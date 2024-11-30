import pandas as pd
import numpy as np
import random

class Winsorizer:
    def __init__(self, lower_percentile=0.05, upper_percentile=0.95):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.column_bounds = {}

    def fit(self, dataframe, columns):
        for column in columns:
            lower_bound = dataframe[column].quantile(self.lower_percentile)
            upper_bound = dataframe[column].quantile(self.upper_percentile)
            self.column_bounds[column] = (lower_bound, upper_bound)

    def transform(self, dataframe):
        winsorized_dataframe = dataframe.copy()
        for column, (lower_bound, upper_bound) in self.column_bounds.items():
            winsorized_dataframe[column] = dataframe[column].apply(
                lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x
            )
        return winsorized_dataframe

class DataOversampler:
    def __init__(self, strategy='simple', num_simulations=5, lower_percentile=0.05,
                 upper_percentile=0.95, discount_factor=0.99):
        """
        Initialize the Monte Carlo Oversampler.

        Parameters:
        -----------
        strategy : str
            The simulation strategy to use. Options are:
            - 'simple': Basic Monte Carlo simulation
            - 'bsm': Black-Scholes-Merton inspired simulation
            - 'dynamic': Dynamic programming based simulation
        num_simulations : int
            Number of Monte Carlo simulations to run
        lower_percentile : float
            Lower percentile for winsorization
        upper_percentile : float
            Upper percentile for winsorization
        discount_factor : float
            Discount factor for dynamic programming approaches
        """
        if strategy not in ['monte_carlo', 'bsm', 'dynamic']:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.strategy = strategy.lower()
        self.num_simulations = num_simulations
        self.target_name = None
        self.winsorizer = Winsorizer(lower_percentile, upper_percentile)
        self.discount_factor = discount_factor
        np.random.seed(123)
        random.seed(123)

    def fit_resample(self, data_frame, target_set):
        self.target_name = target_set.name
        data_frame = pd.concat([data_frame, target_set], axis=1)

        label_in_majority, label_in_minority = self.calculate_labels(data_frame)
        number_of_majority_labels, number_of_minority_labels = self.count_labels(data_frame)
        difference_num_majority_minority_labels = number_of_majority_labels - number_of_minority_labels

        # Fit Winsorizer on original minority class data
        original_minority_data = data_frame[data_frame[self.target_name] == label_in_minority]
        float_columns, categorical_columns = self.identify_columns(original_minority_data)
        self.winsorizer.fit(original_minority_data, float_columns)

        df_minority = self.select_minority_rows(data_frame, label_in_minority)
        df_sampled = self.apply_monte_carlo_simulation(df_minority, float_columns, categorical_columns)

        # Apply Winsorization to augmented data of the minority class
        df_sampled[float_columns] = self.winsorizer.transform(df_sampled[float_columns])

        df_sampled = self.remove_duplicates(df_sampled, difference_num_majority_minority_labels, label_in_minority)
        train_df = self.concatenate_dataframes(data_frame, df_sampled)

        y = train_df[self.target_name]

        return train_df.drop(self.target_name, axis=1), y

    def identify_columns(self, data_frame):
        # Identify float columns with more than 30 unique values
        float_columns = data_frame.select_dtypes(include=['float']).columns
        selected_columns = [col for col in float_columns if data_frame[col].nunique() > 30]

        # Identify categorical columns
        categorical_columns = data_frame.select_dtypes(include=['object', 'category']).columns

        # Identify numerical columns with unique values <= 30 and treat them as categorical
        num_columns = data_frame.select_dtypes(include=['int', 'float']).columns
        small_num_columns = [col for col in num_columns if data_frame[col].nunique() <= 30]
        categorical_columns = categorical_columns.union(small_num_columns)

        return selected_columns, categorical_columns

    def calculate_labels(self, data_frame):
        label_in_majority = data_frame[self.target_name].value_counts().idxmax()
        label_in_minority = data_frame[self.target_name].value_counts().idxmin()
        return label_in_majority, label_in_minority

    def count_labels(self, data_frame):
        number_of_majority_labels = data_frame[self.target_name].value_counts().max()
        number_of_minority_labels = data_frame[self.target_name].value_counts().min()
        return number_of_majority_labels, number_of_minority_labels

    def select_minority_rows(self, data_frame, label_in_minority):
        df_minority = data_frame[data_frame[self.target_name] == label_in_minority]
        return df_minority

    def apply_monte_carlo_simulation(self, df_minority, float_columns, categorical_columns):
        df_sampled = pd.DataFrame()
        for _ in range(self.num_simulations):
            if self.strategy == 'monte_carlo':
                df_simulated = self.monte_carlo_oversampler(df_minority, float_columns, categorical_columns)
            elif self.strategy == 'bsm':
                df_simulated = self.bsm_oversampler(df_minority, float_columns, categorical_columns)
            else:  # dynamic
                df_simulated = self.dynamic_induction_oversampler(df_minority, float_columns, categorical_columns)
            df_sampled = pd.concat([df_sampled, df_simulated], axis=0).reset_index(drop=True)
        return df_sampled

    def monte_carlo_oversampler(self, df, float_columns, categorical_columns):
        simulated_data = {}
        for col in float_columns:
            mean = df[col].mean()
            std = df[col].std()
            simulated_data[col] = np.random.normal(mean, std, size=len(df))

        # Preserve categorical columns from original data
        for col in categorical_columns:
            simulated_data[col] = df[col].values

        return pd.DataFrame(simulated_data)

    def bsm_oversampler(self, df, float_columns, categorical_columns):
        m = len(df)
        simulated_data = pd.DataFrame(index=df.index, columns=float_columns + list(categorical_columns))

        for col in float_columns:
            current_values = df[col].values
            mean = df[col].mean()
            volatility = df[col].std() / mean  # Normalized volatility

            # BSM-inspired simulation with drift and volatility
            dt = 1.0 / m
            drift = (mean - 0.5 * volatility**2) * dt
            diffusion = volatility * np.sqrt(dt)

            # Generate random walks
            random_walks = np.exp(
                drift + diffusion * np.random.normal(0, 1, size=m)
            )

            # Apply random walks to generate new values
            simulated_values = current_values * random_walks
            simulated_data[col] = simulated_values

        # Preserve categorical columns
        for col in categorical_columns:
            simulated_data[col] = df[col].values

        return simulated_data

    def dynamic_induction_oversample(self, df, float_columns, categorical_columns):
        m = len(df)
        V = np.zeros((m, len(float_columns)))
        simulated_data = pd.DataFrame(index=df.index, columns=float_columns + list(categorical_columns))

        # Initialize terminal values
        for idx, col in enumerate(float_columns):
            V[:, idx] = df[col].values

        # Backward induction
        for i in range(m-2, -1, -1):
            for idx, col in enumerate(float_columns):
                mean = df[col].mean()
                std = df[col].std()

                # Generate multiple paths for better expectation estimation
                num_paths = 10
                future_values = np.random.normal(mean, std, size=num_paths)

                # Calculate immediate and continuation values
                immediate_value = V[i+1, idx]
                continuation_value = self.discount_factor * np.mean(future_values)

                # Optimal value and decision
                if immediate_value > continuation_value:
                    V[i, idx] = immediate_value
                    simulated_data.at[df.index[i], col] = V[i+1, idx]
                else:
                    V[i, idx] = continuation_value
                    simulated_data.at[df.index[i], col] = np.random.choice(future_values)

        # Fill any remaining NaN values
        for col in float_columns:
            mean = df[col].mean()
            std = df[col].std()
            mask = simulated_data[col].isna()
            simulated_data.loc[mask, col] = np.random.normal(mean, std, size=mask.sum())

        # Preserve categorical columns
        for col in categorical_columns:
            simulated_data[col] = df[col].values

        return simulated_data

    def remove_duplicates(self, df_sampled, difference_num_majority_minority_labels, label_in_minority):
        df_sampled = df_sampled.drop_duplicates()[:difference_num_majority_minority_labels]
        df_sampled[self.target_name] = label_in_minority
        return df_sampled

    def concatenate_dataframes(self, data_frame, df_sampled):
        train_df = pd.concat([data_frame, df_sampled], axis=0).reset_index(drop=True)
        return train_df
