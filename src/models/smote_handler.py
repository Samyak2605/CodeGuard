from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter
import numpy as np

class SMOTEHandler:
    def __init__(self, strategy='auto', random_state=42):
        """
        Initialize SMOTE handler
        
        Args:
            strategy: 'auto', 'minority', or dict specifying ratios
            random_state: For reproducibility
        """
        self.strategy = strategy
        self.random_state = random_state
        self.smote = None
    
    def analyze_imbalance(self, y):
        """Analyze class distribution"""
        counter = Counter(y)
        total = len(y)
        
        print(f"\nClass Distribution:")
        print(f"  Positive (1): {counter.get(1, 0)} ({counter.get(1, 0)/total*100:.2f}%)")
        print(f"  Negative (0): {counter.get(0, 0)} ({counter.get(0, 0)/total*100:.2f}%)")
        
        if len(counter) == 2:
            majority = max(counter.values())
            minority = min(counter.values())
            ratio = majority / minority if minority > 0 else float('inf')
            print(f"  Imbalance Ratio: {ratio:.2f}:1")
            
            if ratio > 3:
                print(f"  ⚠️ SEVERE IMBALANCE detected! SMOTE recommended.")
                return True, ratio
            elif ratio > 1.5:
                print(f"  ⚠️ Moderate imbalance. SMOTE may help.")
                return True, ratio
            else:
                print(f"  ✅ Relatively balanced. SMOTE not needed.")
                return False, ratio
        
        return False, 1.0
    
    def apply_smote(self, X_train, y_train, sampling_strategy='auto'):
        """
        Apply SMOTE to training data
        """
        
        print(f"\n{'='*60}")
        print(f"APPLYING SMOTE")
        print(f"{'='*60}")
        
        # Check if SMOTE is needed
        needs_smote, ratio = self.analyze_imbalance(y_train)
        
        if not needs_smote and sampling_strategy == 'auto':
            print("✅ Data is balanced. Skipping SMOTE.")
            return X_train, y_train
        
        # Create SMOTE instance
        self.smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
            k_neighbors=5
        )
        
        # Apply SMOTE
        print(f"\nBefore SMOTE:")
        print(f"  Shape: {X_train.shape}")
        self.analyze_imbalance(y_train)
        
        try:
            X_resampled, y_resampled = self.smote.fit_resample(X_train, y_train)
            
            print(f"\nAfter SMOTE:")
            print(f"  Shape: {X_resampled.shape}")
            self.analyze_imbalance(y_resampled)
            
            print(f"\n✅ SMOTE applied successfully!")
            print(f"  Added {len(X_resampled) - len(X_train)} synthetic samples")
            
            return X_resampled, y_resampled
        except ValueError as e:
            print(f"⚠️ SMOTE failed (likely too few samples): {e}")
            return X_train, y_train
    
    def apply_smote_with_undersampling(self, X_train, y_train):
        """
        Apply SMOTE + Random Undersampling (recommended for severe imbalance)
        """
        
        print(f"\n{'='*60}")
        print(f"APPLYING SMOTE + UNDERSAMPLING")
        print(f"{'='*60}")
        
        # Define resampling pipeline
        # Minority to 50% of majority
        over = SMOTE(sampling_strategy=0.5, random_state=self.random_state)
        # Majority to 80% after SMOTE (undersample majority)
        under = RandomUnderSampler(sampling_strategy=0.8, random_state=self.random_state)
        
        pipeline = ImbPipeline(steps=[('over', over), ('under', under)])
        
        print(f"\nBefore resampling:")
        self.analyze_imbalance(y_train)
        
        try:
            X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
            
            print(f"\nAfter resampling:")
            self.analyze_imbalance(y_resampled)
            
            return X_resampled, y_resampled
        except ValueError as e:
            print(f"⚠️ Resampling failed: {e}")
            return X_train, y_train
