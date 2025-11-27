import os
import ast
import time
import logging
import pandas as pd
from github import Github, GithubException
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List, Dict, Optional
import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GithubCollector:
    def __init__(self, output_dir: str = "data/raw", batch_size: int = 10):
        """
        Initialize the GitHub Data Collector.
        
        Args:
            output_dir (str): Directory to save collected data.
            batch_size (int): Number of repositories to process before saving.
        """
        load_dotenv()
        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN not found in environment variables.")
        
        self.g = Github(self.token)
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.output_file = os.path.join(output_dir, "github_functions.csv")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load existing progress if available
        self.processed_repos = self._load_processed_repos()
        
    def _load_processed_repos(self) -> set:
        """Load the set of already processed repositories to support resume capability."""
        if os.path.exists(self.output_file):
            try:
                df = pd.read_csv(self.output_file)
                if 'repo_name' in df.columns:
                    return set(df['repo_name'].unique())
            except Exception as e:
                logger.error(f"Error loading existing data: {e}")
        return set()

    def search_repos(self, query: str = "language:python stars:>100", limit: int = 200) -> List[str]:
        """
        Search for Python repositories on GitHub.
        
        Args:
            query (str): GitHub search query.
            limit (int): Maximum number of repositories to return.
            
        Returns:
            List[str]: List of repository full names (e.g., "owner/repo").
        """
        logger.info(f"Searching for repositories with query: '{query}'")
        repos = []
        try:
            results = self.g.search_repositories(query=query, sort="stars", order="desc")
            count = 0
            for repo in results:
                if count >= limit:
                    break
                if repo.full_name not in self.processed_repos:
                    repos.append(repo.full_name)
                    count += 1
                    # Rate limit handling for search
                    if count % 10 == 0:
                        time.sleep(2)
            logger.info(f"Found {len(repos)} new repositories to process.")
            return repos
        except Exception as e:
            logger.error(f"Error searching repositories: {e}")
            return []

    def extract_functions_from_code(self, code: str, file_path: str, repo_name: str) -> List[Dict]:
        """
        Parse Python code and extract function definitions using AST.
        
        Args:
            code (str): The Python source code.
            file_path (str): Path to the file.
            repo_name (str): Name of the repository.
            
        Returns:
            List[Dict]: List of extracted functions with metadata.
        """
        functions = []
        try:
            tree = ast.parse(code)
            lines = code.splitlines()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Filter: Skip trivial functions (< 5 lines)
                    func_lines = node.end_lineno - node.lineno + 1
                    if func_lines < 5 or func_lines > 300:
                        continue
                        
                    # Extract function source code
                    func_code = "\n".join(lines[node.lineno-1 : node.end_lineno])
                    
                    functions.append({
                        "repo_name": repo_name,
                        "file_path": file_path,
                        "function_name": node.name,
                        "code": func_code,
                        "num_lines": func_lines
                    })
        except SyntaxError:
            pass # Ignore syntax errors in source files
        except Exception as e:
            logger.debug(f"Error parsing file {file_path}: {e}")
            
        return functions

    def process_repo(self, repo_name: str) -> List[Dict]:
        """
        Process a single repository: download files and extract functions.
        
        Args:
            repo_name (str): Repository full name.
            
        Returns:
            List[Dict]: Extracted functions.
        """
        extracted_data = []
        try:
            repo = self.g.get_repo(repo_name)
            contents = repo.get_contents("")
            
            # Queue for BFS traversal of repo contents
            queue = []
            if isinstance(contents, list):
                queue.extend(contents)
            else:
                queue.append(contents)
                
            file_count = 0
            while queue and file_count < 100: # Limit files per repo to avoid massive repos
                file_content = queue.pop(0)
                
                if file_content.type == "dir":
                    try:
                        queue.extend(repo.get_contents(file_content.path))
                    except:
                        pass
                elif file_content.path.endswith(".py"):
                    try:
                        # Decode file content
                        raw_code = file_content.decoded_content.decode("utf-8")
                        functions = self.extract_functions_from_code(raw_code, file_content.path, repo_name)
                        extracted_data.extend(functions)
                        file_count += 1
                    except Exception as e:
                        logger.debug(f"Error reading file {file_content.path}: {e}")
                        
            logger.info(f"Processed {repo_name}: Extracted {len(extracted_data)} functions from {file_count} files.")
            
        except Exception as e:
            logger.error(f"Error processing repo {repo_name}: {e}")
            
        return extracted_data

    def save_data(self, data: List[Dict]):
        """
        Save extracted data to CSV. Appends if file exists.
        
        Args:
            data (List[Dict]): Data to save.
        """
        if not data:
            return
            
        df = pd.DataFrame(data)
        
        # Check if file exists to determine if we need to write header
        header = not os.path.exists(self.output_file)
        
        df.to_csv(self.output_file, mode='a', header=header, index=False)
        logger.info(f"Saved {len(data)} functions to {self.output_file}")

    def run(self, target_repos: int = 200):
        """
        Main execution loop.
        
        Args:
            target_repos (int): Number of repositories to process.
        """
        logger.info("Starting data collection...")
        
        # Search for repos
        repos_to_process = self.search_repos(limit=target_repos)
        
        batch_data = []
        total_functions = 0
        
        progress_bar = tqdm(repos_to_process, desc="Processing Repositories")
        
        for i, repo_name in enumerate(progress_bar):
            # Rate limiting: Sleep slightly between repos
            time.sleep(1)
            
            functions = self.process_repo(repo_name)
            batch_data.extend(functions)
            total_functions += len(functions)
            
            progress_bar.set_postfix({"Functions": total_functions})
            
            # Save in batches
            if (i + 1) % self.batch_size == 0:
                self.save_data(batch_data)
                batch_data = []
                
        # Save remaining data
        if batch_data:
            self.save_data(batch_data)
            
        logger.info(f"Data collection complete. Total functions collected: {total_functions}")

if __name__ == "__main__":
    # Example usage
    try:
        collector = GithubCollector()
        collector.run(target_repos=10) # Run small batch for testing
    except ValueError as e:
        logger.error(str(e))
        print("Please ensure GITHUB_TOKEN is set in .env file")
