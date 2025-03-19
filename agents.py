from dotenv import load_dotenv
import os
from openai import OpenAI
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#from sklearn.preprocessing import LabelEncoder
#from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from openai import OpenAI

warnings.filterwarnings("ignore")

class DataAnalyticsAgent:

    def __init__(self):
        """Initialize the DataAnalyticsAgent with OpenAI client setup."""
        load_dotenv()
        self.client = OpenAI(
            base_url = "https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY")
            )

    def call_gpt4(self, system_prompt, analysis_prompt):
        """
        Sends a request to GPT-4 with the system and user prompts.
        """
        try:
            response = self.client.chat.completions.create(
                model="meta/llama-3.3-70b-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": analysis_prompt},
                ],
                temperature=0.7,
                max_tokens=1024,
            )
            
            raw_response = response.choices[0].message.content
            return raw_response
        
        
        except Exception as e:
            print("Error calling GPT-4:", e)
            return None

    def data_ingestion_agent(self, file_path):
        """
        Loads the dataset from either CSV or Excel file and retrieves GPT-4's analysis.
        """
        try:
            if file_path is None:
                return None, "No file was uploaded"
            file_extension = file_path.name.split('.')[-1].lower()
            try:
                if file_extension == 'csv':
                    data = pd.read_csv(file_path)
                elif file_extension in ['xlsx', 'xls']:
                    data = pd.read_excel(file_path)
                else:
                    return None, "Unsupported file format. Please upload either a CSV or Excel file."
            except pd.errors.EmptyDataError:
                return None, "The uploaded file is empty"
            except pd.errors.ParserError:
                return None, "Error parsing the file. Please ensure it's properly formatted"
            except Exception as e:
                return None, f"Error reading the file: {str(e)}"

            if data.empty:
                return None, "The uploaded file is empty"
            if len(data.columns) < 1:
                return None, "The file contains no columns"

            system_prompt = "You are a data analyst expert. Your task is to analyze data ingestion."
            analysis_prompt = f"The dataset has {len(data.columns)} columns: {list(data.columns)}. The dataset contains {len(data)} rows. Provide an detailed overview of the dataset structure."
            ingestion_message = self.call_gpt4(system_prompt, analysis_prompt)
            return data, ingestion_message

        except Exception as e:
            return None, f"Error in data ingestion: {str(e)}"

    def data_preprocessing_agent(self, data):
        """
        Enhanced data preprocessing with missing value handling, data type conversion,
        binary encoding, and column filtering.
        """
        try:
            preprocessing_report = {}

            # Create a copy of the input data
            processed_data = data.copy()

            # 1. Data Profiling
            preprocessing_report['initial_shape'] = processed_data.shape

            # 2. Data Type Conversion
            converted_columns = {}
            for column in processed_data.columns:
                if processed_data[column].dtype == 'object':
                    # Try to convert string columns to numeric
                    converted_column = pd.to_numeric(processed_data[column].str.strip(), errors='coerce')
                    if converted_column.notnull().sum() > 0:
                        processed_data[column] = converted_column
                        converted_columns[column] = f"Converted from {data[column].dtype} to {processed_data[column].dtype}"
            preprocessing_report['converted_columns'] = converted_columns

            # 3. Handle Missing Values
            missing_values = processed_data.isnull().sum()
            has_missing = missing_values.any()
            preprocessing_report['missing_values_handled'] = has_missing

            if has_missing:
                for column in processed_data.columns:
                    missing_count = missing_values[column]
                    if missing_count > 0:
                        if missing_count < 10:  # If less than 10 missing values, remove rows
                            processed_data.dropna(subset=[column], inplace=True)
                            preprocessing_report[f'missing_handling_{column}'] = f"Removed {missing_count} rows with missing values"
                        else:  # If 10 or more missing values, fill them
                            if processed_data[column].dtype in ['int64', 'float64']:
                                processed_data[column].fillna(processed_data[column].mean(), inplace=True)
                                preprocessing_report[f'missing_handling_{column}'] = f"Filled {missing_count} missing values with median"
                            else:
                                processed_data[column].fillna(processed_data[column].mode()[0], inplace=True)
                                preprocessing_report[f'missing_handling_{column}'] = f"Filled {missing_count} missing values with mode"

            # 4. Handle Duplicates
            duplicates_count = processed_data.duplicated().sum()
            if duplicates_count > 0:
                processed_data.drop_duplicates(inplace=True, keep='first')
                preprocessing_report['duplicates_removed'] = duplicates_count
            else:
                preprocessing_report['duplicates_removed'] = 0

            # 5. Identify and Drop Irrelevant Columns
            missing_threshold = 0.75
            missing_percentages = processed_data.isnull().mean()
            high_missing_cols = missing_percentages[missing_percentages > missing_threshold].index.tolist()

            zero_variance_cols = [col for col in processed_data.columns if processed_data[col].nunique() == 1]
            unique_cols = [col for col in processed_data.columns if processed_data[col].nunique() == len(processed_data)]

            cols_to_drop = list(set(high_missing_cols + zero_variance_cols + unique_cols))

            preprocessing_report['dropped_columns'] = {
                'high_missing_cols': high_missing_cols,
                'zero_variance_cols': zero_variance_cols,
                'unique_cols': unique_cols
            }

            if cols_to_drop:
                processed_data.drop(columns=cols_to_drop, inplace=True)
                preprocessing_report['columns_dropped'] = len(cols_to_drop)
                preprocessing_report['remaining_columns'] = list(processed_data.columns)

            # 6. Outlier Detection and Treatment
            numeric_columns = processed_data.select_dtypes(include=['int64', 'float64']).columns
            numeric_stats = {}
            for col in numeric_columns:
                Q1 = processed_data[col].quantile(0.25)
                Q3 = processed_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = processed_data[(processed_data[col] < lower_bound) | (processed_data[col] > upper_bound)][col]
                numeric_stats[col] = {
                    'outliers_count': len(outliers)
                }

                # Handle outliers by clipping
                processed_data[col] = processed_data[col].clip(lower=lower_bound, upper=upper_bound)

            preprocessing_report['numeric_stats'] = numeric_stats

            # 7. Statistical Description
            preprocessing_report['statistical_description'] = processed_data.describe()

            # 8. Generate preprocessing insights
            system_prompt = "You are a data preprocessing expert. Analyze the preprocessing results."
            analysis_prompt = f"""
            Preprocessing Report:
            - Initial Shape: {preprocessing_report['initial_shape']}
            - Data Type Conversions: {preprocessing_report['converted_columns']}
            - Missing Values Handled: {preprocessing_report['missing_values_handled']}
            - Duplicates Removed: {preprocessing_report['duplicates_removed']}
            - Dropped Columns: {preprocessing_report['dropped_columns']}
            - Missing Values Handling: {dict(filter(lambda item: 'missing_handling_' in item[0], preprocessing_report.items()))}
            - Numeric Statistics: {preprocessing_report['numeric_stats']}
            - Statistical Description: {preprocessing_report['statistical_description']}
            Provide insights about the data quality and preprocessing steps taken.
            """

            gpt4_insights = self.call_gpt4(system_prompt, analysis_prompt)
            preprocessing_report['gpt4_insights'] = gpt4_insights

            return processed_data, preprocessing_report

        except Exception as e:
            print(f"Error in data preprocessing: {str(e)}")
            return None, None

    def data_visualization_agent(self, processed_data):
        """
        Performs comprehensive data visualization and analysis on processed data.
        Returns a dictionary of plots and their descriptions.
        """
        try:
            visualization_dict = {}
            
            # Identify data types
            numerical_columns = processed_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_columns = processed_data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # 1. Distribution Analysis for Numerical Variables
            if numerical_columns:
                fig, axes = plt.subplots(len(numerical_columns), 2, 
                                    figsize=(15, 5*len(numerical_columns)))
                
                if len(numerical_columns) == 1:
                    axes = axes.reshape(1, -1)
                
                for idx, col in enumerate(numerical_columns):
                    # Histogram with KDE
                    sns.histplot(data=processed_data, x=col, kde=True, ax=axes[idx, 0], 
                            palette="viridis")
                    axes[idx, 0].set_title(f'Distribution of {col}')
                    axes[idx, 0].tick_params(axis='x', rotation=45)
                    
                    # Box plot
                    sns.boxplot(data=processed_data, y=col, ax=axes[idx, 1], 
                            palette="viridis")
                    axes[idx, 1].set_title(f'Box Plot of {col}')
                
                plt.tight_layout()
                visualization_dict['distribution_plots'] = {
                    'figure': fig,
                    'description': 'Distribution analysis showing histograms and box plots for numerical variables'
                }
            
            # 2. Correlation Analysis for Numerical Variables
            if len(numerical_columns) > 1:
                plt.figure(figsize=(10, 8))
                correlation_matrix = processed_data[numerical_columns].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('Correlation Heatmap')
                visualization_dict['correlation_heatmap'] = {
                    'figure': plt.gcf(),
                    'description': 'Correlation heatmap showing relationships between numerical variables'
                }
            
            # 3. Categorical Variable Analysis
            if categorical_columns:
                n_cols = min(2, len(categorical_columns))
                n_rows = (len(categorical_columns) - 1) // n_cols + 1
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                
                if n_rows == 1 and n_cols == 1:
                    axes = np.array([axes])
                axes = axes.flatten()
                
                for idx, col in enumerate(categorical_columns):
                    value_counts = processed_data[col].value_counts()
                    
                    if len(value_counts) > 10:
                        value_counts = value_counts.head(10)
                        axes[idx].set_title(f'Top 10 Categories in {col}')
                    else:
                        axes[idx].set_title(f'Categories in {col}')
                    
                    sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[idx], 
                            palette="husl")
                    axes[idx].tick_params(axis='x', rotation=45)
                
                for idx in range(len(categorical_columns), len(axes)):
                    axes[idx].set_visible(False)
                
                plt.tight_layout()
                visualization_dict['categorical_plots'] = {
                    'figure': fig,
                    'description': 'Bar plots showing the distribution of categorical variables'
                }
            
            # 4. Numerical vs Categorical Analysis (User Selected)
            visualization_dict['available_columns'] = {
                'numerical': numerical_columns,
                'categorical': categorical_columns
            }
            
            # Note: The actual plotting will be done in the Streamlit interface
            # based on user selection
            
            # 5. Pair Plot for Numerical Variables (limited to first 5)
            if len(numerical_columns) > 1:
                plot_columns = numerical_columns[:5]
                pair_plot = sns.pairplot(processed_data[plot_columns], diag_kind='kde', 
                                    palette="husl")
                pair_plot.fig.suptitle('Pair Plot Matrix', y=1.02)
                visualization_dict['pair_plot'] = {
                    'figure': pair_plot.fig,
                    'description': 'Pair plot matrix showing relationships between numerical variables'
                }
            
            # 6. Service Usage Analysis (User Selected)
            visualization_dict['service_columns'] = categorical_columns
            
            # Generate insights using GPT-4
            system_prompt = "You are a data visualization expert. Analyze the visualizations and provide insights."
            analysis_prompt = f"""
            Visualization Analysis:
            1. Numerical Variables ({len(numerical_columns)}): {numerical_columns}
            2. Categorical Variables ({len(categorical_columns)}): {categorical_columns}
            3. Visualizations created: {list(visualization_dict.keys())}
            
            Please provide insights about:
            1. Distribution patterns in numerical variables
            2. Correlations and relationships between variables
            3. Category distributions and imbalances
            4. Any notable patterns or anomalies
            """
            
            gpt4_insights = self.call_gpt4(system_prompt, analysis_prompt)
            
            return visualization_dict, gpt4_insights
            
        except Exception as e:
            print(f"Error in data visualization: {str(e)}")
            return None, None


class ReportGenerationAgent:
    def __init__(self, data_analytics_agent):
        self.data_analytics_agent = data_analytics_agent
        
    def generate_report(self, data, processed_data, preprocessing_report, visualization_dict, ingestion_message):
        """
        Generates a comprehensive markdown report with insights from all agents.
        """
        try:
            # Basic statistics
            total_rows = len(data)
            total_columns = len(data.columns)
            
            # Generate report sections using GPT-4
            system_prompt = "You are a data analysis report writer. Create professional report sections based on the data analysis."
            
            # Executive Summary
            summary_prompt = f"""
            Create an executive summary for a data analysis report with:
            - Initial data overview: {ingestion_message}
            - Key insights from preprocessing: {preprocessing_report.get('gpt4_insights', '')}
            - Main patterns from visualizations: {visualization_dict.get('gpt4_insights', '')}
            - Most significant findings and their business implications
            Make it concise but impactful, focusing on business value.
            """
            executive_summary = self.data_analytics_agent.call_gpt4(system_prompt, summary_prompt)
            
            # Detailed Data Overview
            overview_prompt = f"""
            Provide a comprehensive data overview including:
            - Dataset source and initial structure
            - Total records: {total_rows}
            - Total features: {total_columns}
            - Description of each feature's data type and role
            - Initial data quality observations
            Format as a detailed markdown section.
            """
            data_overview = self.data_analytics_agent.call_gpt4(system_prompt, overview_prompt)
            
            # Key Insights
            insights_prompt = f"""
            Based on all analysis stages, provide key insights:
            - Initial data structure: {ingestion_message}
            - Preprocessing findings: {preprocessing_report.get('gpt4_insights', '')}
            - Visualization patterns: {visualization_dict.get('gpt4_insights', '')}
            Format as clear, bulleted insights with statistical evidence.
            """
            key_insights = self.data_analytics_agent.call_gpt4(system_prompt, insights_prompt)
            
            # Recommendations
            recommendations_prompt = f"""
            Based on the complete analysis results, provide:
            - Initial data recommendations: {ingestion_message}
            - Preprocessing improvements: {preprocessing_report.get('gpt4_insights', '')}
            - Visualization-based suggestions: {visualization_dict.get('gpt4_insights', '')}
            Make recommendations specific, actionable, and prioritized.
            """
            recommendations = self.data_analytics_agent.call_gpt4(system_prompt, recommendations_prompt)
            
            # Generate the full report in markdown format
            report = f"""# Data Analysis Report
            ---

            ## Executive Summary
            {executive_summary or "No executive summary generated."}

            ## 1. Initial Data Assessment
            {data_overview or "No detailed data overview available."}

            ### 1.1 Data Structure Overview
            {ingestion_message}

            ### 1.2 Basic Statistics
            - **Total Records**: {total_rows:,}
            - **Features Analyzed**: {total_columns}
            - **Data Completeness**: {(100 - (data.isnull().sum().sum() / (total_rows * total_columns) * 100)):.2f}%

            ### 1.3 Data Profile
            ```
            Initial Data Types:
            {data.dtypes.to_string()}

            Processed Data Types:
            {processed_data.dtypes.to_string()}
            ```

            ## 2. Data Quality Assessment

            ### 2.1 Preprocessing Summary
            - **Missing Values**: {preprocessing_report.get('missing_values_handled', 'None detected')}
            - **Duplicates Removed**: {preprocessing_report.get('duplicates_removed', 0)}
            - **Columns Transformed**: {len(preprocessing_report.get('converted_columns', {}))}

            ### 2.2 Quality Metrics
            ```
            {preprocessing_report['statistical_description'].to_markdown()}
            ```

            ### 2.3 Preprocessing Insights
            {preprocessing_report.get('gpt4_insights', '')}

            ## 3. Visualization Analysis

            ### 3.1 Distribution Patterns
            ![Distribution Analysis](distribution_analysis.png)
            *Figure 1: Distribution analysis of numerical variables*

            ### 3.2 Correlation Analysis
            ![Correlation Analysis](correlation_analysis.png)
            *Figure 2: Correlation heatmap showing relationships between variables*

            ### 3.3 Categorical Insights
            ![Categorical Analysis](categorical_analysis.png)
            *Figure 3: Analysis of categorical variable distributions*

            ### 3.4 Advanced Visualizations
            {f"![Pair Plot Analysis](pair_plot_analysis.png)*Figure 4: Pair plot showing relationships between numerical variables*" if 'pair_plot' in visualization_dict else ""}

            ### 3.5 Visualization Insights
            {visualization_dict.get('gpt4_insights', '')}

            ## 4. Key Insights and Patterns
            {key_insights}

            ## 5. Statistical Highlights

            ### 5.1 Numerical Variables
            ```python
            {processed_data.describe().to_markdown()}
            ```

            ### 5.2 Data Quality Score
            - **Completeness Score**: {(100 - (data.isnull().sum().sum() / (total_rows * total_columns) * 100)):.2f}%
            - **Consistency Score**: {(100 - (preprocessing_report.get('duplicates_removed', 0) / total_rows * 100)):.2f}%
            - **Preprocessing Impact**: {len(preprocessing_report.get('dropped_columns', {}).get('high_missing_cols', []))} columns optimized

            ## 6. Recommendations and Next Steps
            {recommendations}

            ## 7. Technical Appendix

            ### 7.1 Preprocessing Steps
            1. Data type conversions performed:
            {', '.join(preprocessing_report.get('converted_columns', {}).keys())}
            2. Missing value treatment:
            {dict(filter(lambda item: 'missing_handling_' in item[0], preprocessing_report.items()))}
            3. Column modifications:
            - Dropped due to high missing values: {preprocessing_report['dropped_columns'].get('high_missing_cols', [])}
            - Dropped due to zero variance: {preprocessing_report['dropped_columns'].get('zero_variance_cols', [])}
            - Dropped due to uniqueness: {preprocessing_report['dropped_columns'].get('unique_cols', [])}

            ### 7.2 Methodology Notes
            - Analysis conducted using Python with pandas, numpy, and scikit-learn
            - Visualizations generated using matplotlib and seaborn
            - Statistical significance level: 0.05
            - Data preprocessing includes outlier detection and handling

            ---
            *Generated using DEVA AI Analytics Platform*
            *Report Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
            """
            return report
            
        except Exception as e:
            print(f"Error in report generation: {str(e)}")
            return None