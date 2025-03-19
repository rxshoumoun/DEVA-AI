import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from agents import DataAnalyticsAgent
import numpy as np
import os
import tempfile
from datetime import datetime
from agents import ReportGenerationAgent
import re

def main():
    agents = DataAnalyticsAgent()

    st.set_page_config(
        page_title="🤖 DEVA AI: AI-Powered Data Analysis", layout="wide")
    st.title("🤖 DEVA AI: AI-Powered Data Analysis")
    st.write("📊 Upload your dataset and get AI-powered insights and analysis.")

    uploaded_file = st.file_uploader(
        "📂 Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Please upload a CSV or Excel file containing your dataset"
    )

    if uploaded_file is not None:
        try:
            # Data Overview Section
            st.header("1. Data Overview")
            data, ingestion_message = agents.data_ingestion_agent(uploaded_file)

            if data is not None:
                st.success("✅ Data successfully loaded!")
                st.write("Dataset Shape:", data.shape)
                st.write("Sample of the dataset:")
                st.dataframe(data.head())

                # Data Preprocessing Section
                st.header("2. Data Preprocessing:")
                with st.spinner("Preprocessing data..."):
                    try:
                        processed_data, preprocessing_report = agents.data_preprocessing_agent(data)

                        if processed_data is not None:
                            st.success("✅ Data preprocessed successfully!")

                            # Create tabs for different preprocessing sections
                            preprocess_tabs = st.tabs([
                                "Data Types",
                                "Missing Values",
                                "Duplicates & Columns",
                                "Statistics"
                            ])

                            # Data Types Tab
                            with preprocess_tabs[0]:
                                if preprocessing_report['converted_columns']:
                                    st.subheader("Data Type Conversions")
                                    for col, conversion in preprocessing_report['converted_columns'].items():
                                        st.write(f"• {col}: {conversion}")
                                st.subheader("Current Data Types")
                                dt_df = pd.DataFrame(processed_data.dtypes, columns=['Data Type']).transpose()
                                st.dataframe(dt_df)

                            # Missing Values Tab
                            with preprocess_tabs[1]:
                                if preprocessing_report.get('missing_values_handled', False):
                                    st.subheader("Missing Values Handling")
                                    missing_handling = {k: v for k, v in preprocessing_report.items()
                                                    if k.startswith('missing_handling_')}
                                    for column, action in missing_handling.items():
                                        st.write(f"• {column.replace('missing_handling_', '')}: {action}")
                                else:
                                    st.info("No missing values found in the dataset.")

                            # Duplicates & Columns Tab
                            with preprocess_tabs[2]:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.subheader("Duplicates Handled")
                                    st.write(f"Total duplicates removed: {preprocessing_report['duplicates_removed']}")

                                with col2:
                                    st.subheader("Column Analysis")
                                    dropped_cols = preprocessing_report['dropped_columns']
                                    if any(dropped_cols.values()):
                                        if dropped_cols['high_missing_cols']:
                                            st.write("High missing values:", dropped_cols['high_missing_cols'])
                                        if dropped_cols['zero_variance_cols']:
                                            st.write("Zero variance:", dropped_cols['zero_variance_cols'])
                                        if dropped_cols['unique_cols']:
                                            st.write("Unique values:", dropped_cols['unique_cols'])
                                    else:
                                        st.info("No columns were dropped during preprocessing.")

                            # Statistics Tab
                            with preprocess_tabs[3]:
                                st.subheader("Statistical Description")
                                st.dataframe(preprocessing_report['statistical_description'])

                            # Show processed data sample in expandable section
                            with st.expander("View Processed Data Sample"):
                                st.dataframe(processed_data.head())

                        else:
                            st.error("❌ Error occurred during preprocessing")
                    except Exception as e:
                        st.error(f"❌ Preprocessing error: {str(e)}")
                        processed_data = None

                # Data Visualization Section         
                if processed_data is not None:
                    st.header("3. Data Visualization")
                    with st.spinner("Generating visualizations..."):
                        try:
                            visualization_dict, visualization_insights = agents.data_visualization_agent(processed_data)

                            if visualization_dict and visualization_insights:
                                viz_tabs = st.tabs([
                                    "📊 Distribution Analysis",
                                    "🔄 Correlation Analysis",
                                    "📋 Categorical Analysis",
                                    "📈 Numerical vs Categorical",
                                    "🔍 Pair Plot Analysis",
                                    "🔧 Service Usage Analysis"
                                ])

                                # Distribution Analysis Tab
                                with viz_tabs[0]:
                                    if 'distribution_plots' in visualization_dict:
                                        st.subheader("Distribution Analysis")
                                        st.write(visualization_dict['distribution_plots']['description'])
                                        st.pyplot(visualization_dict['distribution_plots']['figure'])
                                    else:
                                        st.info("No numerical variables available for distribution analysis.")

                                # Correlation Analysis Tab
                                with viz_tabs[1]:
                                    if 'correlation_heatmap' in visualization_dict:
                                        st.subheader("Correlation Heatmap")
                                        st.write(visualization_dict['correlation_heatmap']['description'])
                                        st.pyplot(visualization_dict['correlation_heatmap']['figure'])
                                    else:
                                        st.info("Insufficient numerical variables for correlation analysis.")

                                # Categorical Analysis Tab
                                with viz_tabs[2]:
                                    if 'categorical_plots' in visualization_dict:
                                        st.subheader("Categorical Variable Analysis")
                                        st.write(visualization_dict['categorical_plots']['description'])
                                        st.pyplot(visualization_dict['categorical_plots']['figure'])
                                    else:
                                        st.info("No categorical variables found in the dataset.")

                                # Numerical vs Categorical Tab
                                with viz_tabs[3]:
                                    st.subheader("Numerical vs Categorical Analysis")
                                    if 'available_columns' in visualization_dict:
                                        # User selection for columns
                                        selected_num_cols = st.multiselect(
                                            "Select numerical columns (max 4):",
                                            visualization_dict['available_columns']['numerical'],
                                            max_selections=4
                                        )
                                        selected_cat_cols = st.multiselect(
                                            "Select categorical columns (max 4):",
                                            visualization_dict['available_columns']['categorical'],
                                            max_selections=4
                                        )

                                        if selected_num_cols and selected_cat_cols:
                                            fig, axes = plt.subplots(len(selected_num_cols), len(selected_cat_cols),
                                                                figsize=(15, 5*len(selected_num_cols)))
                                            
                                            if len(selected_num_cols) == 1 and len(selected_cat_cols) == 1:
                                                axes = np.array([[axes]])
                                            elif len(selected_num_cols) == 1 or len(selected_cat_cols) == 1:
                                                axes = axes.reshape(-1, 1)
                                            
                                            for i, num_col in enumerate(selected_num_cols):
                                                for j, cat_col in enumerate(selected_cat_cols):
                                                    sns.boxplot(data=processed_data, x=cat_col, y=num_col, 
                                                            ax=axes[i, j], palette="husl")
                                                    axes[i, j].set_title(f'{num_col} by {cat_col}')
                                                    axes[i, j].tick_params(axis='x', rotation=45)
                                            
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                    else:
                                        st.info("No suitable variables available for this analysis.")

                                # Pair Plot Analysis Tab
                                with viz_tabs[4]:
                                    if 'pair_plot' in visualization_dict:
                                        st.subheader("Pair Plot Analysis")
                                        st.write(visualization_dict['pair_plot']['description'])
                                        st.pyplot(visualization_dict['pair_plot']['figure'])
                                    else:
                                        st.info("Insufficient numerical variables for pair plot analysis.")

                                # Service Usage Analysis Tab
                                with viz_tabs[5]:
                                    st.subheader("Service Usage Analysis")
                                    if 'service_columns' in visualization_dict:
                                        selected_service_cols = st.multiselect(
                                            "Select service columns to analyze:",
                                            visualization_dict['service_columns']
                                        )

                                        if selected_service_cols:
                                            # Create service usage visualization
                                            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                                            axes = axes.flatten()

                                            for idx, col in enumerate(selected_service_cols[:4]):  # Limit to 4 plots
                                                sns.countplot(data=processed_data, x=col, ax=axes[idx], 
                                                            palette="Set3")
                                                axes[idx].set_title(f'Usage Distribution: {col}')
                                                axes[idx].tick_params(axis='x', rotation=45)

                                            # Hide empty subplots if less than 4 services selected
                                            for idx in range(len(selected_service_cols[:4]), 4):
                                                axes[idx].set_visible(False)

                                            plt.tight_layout()
                                            st.pyplot(fig)

                                            # Display service usage metrics
                                            with st.expander("View Service Usage Metrics"):
                                                for col in selected_service_cols:
                                                    st.subheader(f"{col} Usage Breakdown")
                                                    usage_counts = processed_data[col].value_counts()
                                                    usage_percentages = usage_counts / len(processed_data) * 100
                                                    
                                                    metrics_df = pd.DataFrame({
                                                        'Count': usage_counts,
                                                        'Percentage': usage_percentages
                                                    })
                                                    st.dataframe(metrics_df)
                                    else:
                                        st.info("No service columns available for analysis.")

                            else:
                                st.warning("No visualizations could be generated for this dataset.")

                        except Exception as e:
                            st.error(f"Error generating visualizations: {str(e)}")
                

                # Update the report generation section in the main() function
                if visualization_dict is not None:
                    st.header("4. Report Generation")
                    report_agent = ReportGenerationAgent(agents)

                    with st.spinner("Generating comprehensive report..."):
                        try:
                            # Temporary directory for visualizations
                            with tempfile.TemporaryDirectory() as temp_dir:
                                # Save visualization images
                                report_agent.save_visualizations(visualization_dict, temp_dir)
                                
                                # Generate complete report
                                report_content = report_agent.generate_report(
                                    data, 
                                    processed_data, 
                                    preprocessing_report, 
                                    visualization_dict,
                                    ingestion_message
                                )
                                
                                # Report Preview and Download
                                if report_content:
                                    st.success("✅ Report generated successfully!")
                                    
                                    # Tabs for different report sections
                                    report_tabs = st.tabs([
                                        "📋 Executive Summary", 
                                        "📊 Data Overview", 
                                        "🔍 Statistical Highlights", 
                                        "📈 Visualizations", 
                                        "🎯 Recommendations"
                                    ])

                                    # Executive Summary Tab
                                    with report_tabs[0]:
                                        st.subheader("Executive Summary")
                                        try:
                                            # Find the executive summary section more precisely
                                            summary_start = report_content.find("## Executive Summary")
                                            summary_end = report_content.find("## 1.", summary_start)
                                            if summary_start != -1 and summary_end != -1:
                                                summary_text = report_content[summary_start:summary_end].replace("## Executive Summary", "").strip()
                                                st.markdown(summary_text)
                                            else:
                                                st.write("Executive summary not found.")
                                        except Exception as e:
                                            st.error(f"Error extracting executive summary: {e}")

                                    with report_tabs[1]:
                                        st.subheader("Data Overview")
                                        try:
                                            # More robust parsing of the data overview section
                                            match = re.search(r'## 1\. Initial Data Assessment(.*?)## 2\.',
                                                            report_content, 
                                                            re.DOTALL | re.MULTILINE | re.IGNORECASE)
                                            
                                            if match:
                                                overview_text = match.group(1).strip()
                                                # Remove any leading or trailing whitespace
                                                overview_text = overview_text.strip()
                                                
                                                # If the overview is empty, provide a fallback
                                                if not overview_text:
                                                    overview_text = "### No detailed overview available\n\nBasic dataset information could not be extracted."
                                                
                                                # Render the markdown
                                                st.markdown(overview_text)
                                            else:
                                                # Fallback if regex fails
                                                st.info("Data overview section could not be found in the report.")
                                        
                                        except Exception as e:
                                            st.error(f"Error extracting data overview: {e}")
                                            # Provide more context about the error
                                            st.write(f"Error details: {str(e)}")
                                            # Optionally, print the full report content for debugging
                                            st.code(report_content)

                                    # Statistical Highlights Tab
                                    with report_tabs[2]:
                                        st.subheader("Statistical Highlights")
                                        try:
                                            # Find the statistical highlights section more precisely
                                            stats_start = report_content.find("## 5. Statistical Highlights")
                                            stats_end = report_content.find("## 6.", stats_start)
                                            if stats_start != -1 and stats_end != -1:
                                                stats_text = report_content[stats_start:stats_end].replace("## 5. Statistical Highlights", "").strip()
                                                st.markdown(stats_text)
                                            else:
                                                st.write("Statistical highlights not found.")
                                        except Exception as e:
                                            st.error(f"Error extracting statistical highlights: {e}")

                                    # Visualizations Tab
                                    with report_tabs[3]:
                                        st.subheader("Data Visualizations")
                                        
                                        # Create columns for visualization display
                                        cols = st.columns(2)
                                        
                                        # Define visualization image names
                                        viz_images = [
                                            'distribution_analysis.png',
                                            'correlation_analysis.png',
                                            'categorical_analysis.png',
                                            'pair_plot_analysis.png'
                                        ]
                                        
                                        # Track if any images were found
                                        images_found = False
                                        
                                        # Display visualizations
                                        for i, img_name in enumerate(viz_images):
                                            img_path = os.path.join(temp_dir, img_name)
                                            if os.path.exists(img_path):
                                                images_found = True
                                                # Alternate between left and right columns
                                                with cols[i % 2]:
                                                    st.image(img_path, caption=f'Visualization: {img_name}', use_column_width=True)
                                        
                                        # If no images were found, show an informative message
                                        if not images_found:
                                            st.warning("No visualization images could be generated. This might be due to insufficient data or processing limitations.")
                                    # Recommendations Tab
                                    with report_tabs[4]:
                                        st.subheader("Recommendations")
                                        try:
                                            # Find the recommendations section more precisely
                                            recommendations_start = report_content.find("## 6. Recommendations and Next Steps")
                                            recommendations_end = report_content.find("## 7.", recommendations_start)
                                            if recommendations_start != -1:
                                                if recommendations_end == -1:
                                                    recommendations_end = len(report_content)
                                                recommendations_text = report_content[recommendations_start:recommendations_end].replace("## 6. Recommendations and Next Steps", "").strip()
                                                st.markdown(recommendations_text)
                                            else:
                                                st.write("Recommendations not found.")
                                        except Exception as e:
                                            st.error(f"Error extracting recommendations: {e}")

                                    # Download functionality
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    report_filename = f"data_analysis_report_{timestamp}.md"
                                    
                                    st.download_button(
                                        label="📥 Download Full Report",
                                        data=report_content,
                                        file_name=report_filename,
                                        mime="text/markdown",
                                    )

                                else:
                                    st.error("❌ Failed to generate report")
                                    
                        except Exception as e:
                            st.error(f"❌ Error generating report: {str(e)}")       
            else:
                st.error(f"Failed to load the dataset: {ingestion_message}")

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
    else:
        st.info("Please upload a CSV or Excel file to begin the analysis.")

    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        🚀 This dashboard provides AI-powered data analysis using:
        - 🤖 Llama-3.1 for insights and suggestions
        - 🔄 Automated data preprocessing
        - 📊 Interactive visualizations
        - 🎯 Machine learning model training
        """)

        st.header("📝 Instructions")
        st.write("""
        1. Upload a CSV or Excel file using the file uploader
        2. Review the data overview and preprocessing steps
        3. Explore the automated visualizations in different tabs
        4. Check AI-generated insights for each analysis
        """)

if __name__ == "__main__":
    main()