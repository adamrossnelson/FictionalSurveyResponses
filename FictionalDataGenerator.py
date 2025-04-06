import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

class MakeData:
    """
    A class for generating fictional survey data with controlled factor loadings.
    
    This class allows you to create simulated survey data with specific factor structures,
    making it useful for testing factor analysis methods or creating demo datasets.
    
    Attributes:
        n_subjects (int): Number of subjects/respondents in the dataset
        factors (Dict): Dictionary of factor configurations
        seed (int): Random seed for reproducibility
        _data (pd.DataFrame): Generated data (only available after run() is called)
    """
    
    def __init__(self, 
                 n_subjects: int = 1000, 
                 seed: Optional[int] = None):
        """
        Initialize the MakeData object.
        
        Args:
            n_subjects (int): Number of subjects/respondents to generate
            seed (Optional[int]): Random seed for reproducibility
        """
        self.n_subjects = n_subjects
        self.factors = {}
        self.seed = seed
        self._data = None
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
    
    def add_factor(self, 
                  name: str,
                  n_items: int = 4, 
                  distribution: Union[List[float], str] = "normal",
                  mean: float = 3, 
                  std: float = 1,
                  min_val: int = 1,
                  max_val: int = 5,
                  noise_range: List = [-2, -1, 0, 0, 0, 1, 1, 2]):
        """
        Add a factor to the data generation model.
        
        Args:
            name (str): Name of the factor
            n_items (int): Number of survey items to generate for this factor
            distribution (Union[List[float], str]): Either a list of probabilities for values 1-5,
                                                  or "normal" for normal distribution
            mean (float): Mean value if using normal distribution
            std (float): Standard deviation if using normal distribution
            min_val (int): Minimum value for survey responses
            max_val (int): Maximum value for survey responses
            noise_range (List): Possible noise values to add to the base factor
        """
        self.factors[name] = {
            'n_items': n_items,
            'distribution': distribution,
            'mean': mean,
            'std': std,
            'min_val': min_val,
            'max_val': max_val,
            'noise_range': noise_range
        }
        
        return self  # Allow method chaining
    
    def _generate_factor_values(self, factor_config: Dict) -> np.ndarray:
        """
        Generate base factor values according to the specified distribution.
        
        Args:
            factor_config (Dict): Factor configuration dictionary
            
        Returns:
            np.ndarray: Generated factor values
        """
        dist = factor_config['distribution']
        
        if isinstance(dist, list) and len(dist) == (factor_config['max_val'] - factor_config['min_val'] + 1):
            # Use provided probability distribution
            values = np.random.choice(
                np.arange(factor_config['min_val'], factor_config['max_val'] + 1), 
                size=self.n_subjects, 
                p=dist
            )
            
        elif dist == "normal":
            # Generate from normal distribution and clip to min/max
            values = np.random.normal(
                factor_config['mean'], 
                factor_config['std'], 
                self.n_subjects
            )
            values = np.clip(
                np.round(values), 
                factor_config['min_val'], 
                factor_config['max_val']
            )
        else:
            raise ValueError(f"Unknown distribution: {dist}. " 
                            f"Use 'normal' or provide a list of probabilities.")
            
        return values
    
    def _generate_items(self, factor_values: np.ndarray, factor_name: str, factor_config: Dict) -> Dict[str, np.ndarray]:
        """
        Generate survey items based on factor values plus noise.
        
        Args:
            factor_values (np.ndarray): Base factor values
            factor_name (str): Name of the factor
            factor_config (Dict): Factor configuration dictionary
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of item names and their values
        """
        items = {}
        
        for i in range(1, factor_config['n_items'] + 1):
            item_name = f"{factor_name}_{i}"
            
            # Add noise to the factor values
            noise = np.random.choice(factor_config['noise_range'], size=self.n_subjects)
            item_values = factor_values + noise
            
            # Map to valid range (1-5 typically)
            valid_values = np.clip(item_values, 
                                   factor_config['min_val'], 
                                   factor_config['max_val'])
            
            items[item_name] = valid_values
            
        return items

    def run(self) -> pd.DataFrame:
        """
        Generate the survey data based on the configured factors.
        
        Returns:
            pd.DataFrame: Generated survey data
        """
        if not self.factors:
            raise ValueError("No factors defined. Use add_factor() method to add factors.")
        
        # Initialize dataframe with subject IDs
        df = pd.DataFrame({'subject_id': range(1, self.n_subjects + 1)})
        
        # Generate data for each factor
        for factor_name, factor_config in self.factors.items():
            # Generate base factor values
            factor_values = self._generate_factor_values(factor_config)
            
            # Add the factor itself to the dataframe with a _raw suffix to distinguish from item names
            df[f"{factor_name}_raw"] = factor_values
            
            # Generate items based on the factor
            items = self._generate_items(factor_values, factor_name, factor_config)
            
            # Add items to the dataframe
            for item_name, item_values in items.items():
                df[item_name] = item_values
        
        self._data = df
        return df
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the generated data. Must call run() first.
        
        Returns:
            pd.DataFrame: Generated survey data
        """
        if self._data is None:
            raise ValueError("Data not generated yet. Call run() first.")
        return self._data
