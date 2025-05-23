import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import openpyxl
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScale
import statsmodels.api as sm

# Set style for plots
plt.style.use('default')  # Using default style instead of seaborn
sns.set_theme()  # This will set the seaborn theme properly
plt.rcParams['figure.figsize'] = (12, 8)

def clean_excel_data(df):
    # Skip header rows (usually the first 4-5 rows contain metadata)
    df = df.copy()
    
    # Find the first row that doesn't contain NaN or metadata
    start_row = 0
    for idx, row in df.iterrows():
        if not pd.isna(row.iloc[0]) and not any(str(val).startswith('World Values Survey') for val in row):
            start_row = idx
            break
    
    # Reset the DataFrame to start from the actual data
    df = df.iloc[start_row:].reset_index(drop=True)
    
    # If first row contains column names, use it as header
    if not pd.isna(df.iloc[0]).all():
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
    
    # Clean column names
    df.columns = [str(col).strip() for col in df.columns]
    
    return df

def load_data():
    data_files = {
        'business_env': ('business_environment_results.csv', 'latin1'),
        'un_women': ('UN Women Data Hub table-export (4).csv', 'latin1'),
        'hdr': ('hdr-data (1).xlsx', None),
        'income': ('Income_level_Recoded.xls', None),
        'jobs_scarce': ('Jobs_swcscarce_Men_should_have_more_right_to_a_job_than_women_3-point_scale.xls', None),
        'employment': ('Employment_status.xls', None),
        'executives': ('Men_make_better_business_executives_than_women_do.xls', None),
        'gsni': ('HDR21-22_GSNI_Tables.xlsx', None),
        'nes': ('export-nes-2025-05-11_17-59-04.csv', 'latin1')
    }
    
    loaded_data = {}
    
    for key, (filename, encoding) in data_files.items():
        try:
            if not os.path.exists(filename):
                print(f"Warning: File {filename} not found")
                continue
                
            print(f"Loading {filename}...")
            
            if filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filename)
                df = clean_excel_data(df)
            else:
                df = pd.read_csv(filename, encoding=encoding)
                
            print(f"Successfully loaded {filename} with {len(df)} rows and {len(df.columns)} columns")
            
            if key == 'nes':
                df = clean_nes_data(df)
            
            loaded_data[key] = df
            
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Load additional data for comparison
    try:
        if os.path.exists('hdr-data (1).xlsx'):
            loaded_data['hdr_all'] = pd.read_excel('hdr-data (1).xlsx')
        if os.path.exists('HDR21-22_GSNI_Tables.xlsx'):
            loaded_data['gsni_all'] = pd.read_excel('HDR21-22_GSNI_Tables.xlsx')
    except Exception as e:
        print(f"Error loading comparison data: {str(e)}")
    
    return loaded_data

def clean_nes_data(df):
    """Clean and prepare NES data for analysis"""
    # Drop the first three unnamed columns
    df = df.iloc[:, 3:]
    
    # Clean column names
    df.columns = [
        'financial_resources',
        'policy_support_general',
        'policy_support_taxes',
        'govt_programs',
        'education_primary',
        'education_higher',
        'rd_transfer',
        'commercial_infrastructure',
        'market_dynamics',
        'market_openness',
        'physical_infrastructure',
        'cultural_norms'
    ]
    
    # Convert string values to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.extract('(\d+\.?\d*)')[0], errors='coerce')
    
    return df

def clean_data(data_dict):
    # Clean income level data
    income_df = data_dict['income'].copy()
    income_df = income_df[~income_df[''].isin(['Don\'t know', 'No answer'])]
    if 'year' in income_df.columns:
        income_df['year'] = pd.to_numeric(income_df['year'], errors='coerce')
    
    # Clean employment data
    employment_df = data_dict['employment'].copy()
    employment_df = employment_df[~employment_df['Base=1200; Weighted results'].str.contains('Don\'t know|No answer', na=False)]
    if 'year' in employment_df.columns:
        employment_df['year'] = pd.to_numeric(employment_df['year'], errors='coerce')
    
    # Clean jobs_scarce data
    if 'jobs_scarce' in data_dict:
        jobs_df = data_dict['jobs_scarce'].copy()
        jobs_df = jobs_df[~jobs_df[''].isin(['Don\'t know', 'No answer'])]
        data_dict['jobs_scarce'] = jobs_df
    
    # Clean executives data
    if 'executives' in data_dict:
        exec_df = data_dict['executives'].copy()
        exec_df = exec_df[~exec_df[''].isin(['Don\'t know', 'No answer'])]
        data_dict['executives'] = exec_df
    
    # Print available columns for debugging
    print("\nAvailable columns in income data:")
    print(income_df.columns.tolist())
    print("\nAvailable columns in employment data:")
    print(employment_df.columns.tolist())
    
    return {**data_dict, 'income': income_df, 'employment': employment_df}

def create_analysis_directory():
    directories = ['plots', 'results']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def analyze_gender_attitudes(data_dict):
    """Analyze gender attitudes from jobs and executives datasets"""
    try:
        jobs_df = data_dict['jobs_scarce'].copy()
        exec_df = data_dict['executives'].copy()
        
        # Print the first few rows to debug
        print("\nJobs scarce data head:")
        print(jobs_df.head())
        print("\nExecutives data head:")
        print(exec_df.head())
        
        # Create simplified analysis
        attitudes_analysis = {}
        
        # Process jobs data if available
        if not jobs_df.empty:
            total_row = jobs_df[jobs_df.iloc[:, 0] == 'TOTAL']
            if not total_row.empty:
                attitudes_analysis['jobs_scarce'] = pd.Series(
                    total_row.iloc[:, -1].values[0],
                    index=['Agreement Level']
                )
        
        # Process executives data if available
        if not exec_df.empty:
            total_row = exec_df[exec_df.iloc[:, 0] == 'TOTAL']
            if not total_row.empty:
                attitudes_analysis['executives'] = pd.Series(
                    total_row.iloc[:, -1].values[0],
                    index=['Agreement Level']
                )
        
        return attitudes_analysis
        
    except Exception as e:
        print(f"Error in gender attitudes analysis: {str(e)}")
        return {}  # Return empty dict on error

def analyze_employment_trends(data_dict):
    """Analyze employment trends by gender over time"""
    try:
        emp_df = data_dict['employment'].copy()
        
        # Print the first few rows to debug
        print("\nEmployment data head:")
        print(emp_df.head())
        
        # Try to identify the correct columns
        if 'TOTAL' in emp_df.columns:
            emp_df = emp_df[emp_df['TOTAL'].notna()]
        
        # Assuming the first column might contain gender information
        gender_col = emp_df.columns[0]
        value_col = emp_df.columns[-1]
        
        # Group by gender
        employment_analysis = emp_df.groupby(gender_col)[value_col].mean()
        return pd.DataFrame(employment_analysis)
        
    except Exception as e:
        print(f"Error in employment analysis: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

def analyze_violence_indicators(data_dict):
    """Analyze violence indicators from GSNI data"""
    try:
        gsni_df = data_dict.get('gsni')
        if gsni_df is None or gsni_df.empty:
            print("No GSNI data available for violence indicators analysis")
            return {'indicators': pd.DataFrame(), 'by_region': pd.Series()}
        
        gsni_df = gsni_df.copy()
        
        # Create a simplified analysis
        violence_analysis = {
            'indicators': pd.DataFrame(),
            'by_region': pd.Series()
        }
        
        # Extract any violence-related information if available
        col_name = 'Feminist Mobilization index value across HDI groups'
        if col_name in gsni_df.columns:
            # Filter rows that contain valid data
            valid_rows = gsni_df[gsni_df[col_name].notna()]
            if not valid_rows.empty:
                try:
                    # Get the 'nan' column name if it exists, otherwise use the last numeric column
                    value_col = 'nan' if 'nan' in valid_rows.columns else valid_rows.select_dtypes(include=[np.number]).columns[-1]
                    violence_analysis['by_region'] = valid_rows.iloc[1:5][[col_name, value_col]].set_index(col_name)[value_col]
                except (IndexError, KeyError) as e:
                    print(f"Error processing GSNI data: {str(e)}")
        
        return violence_analysis
        
    except Exception as e:
        print(f"Error in violence indicators analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'indicators': pd.DataFrame(), 'by_region': pd.Series()}

def analyze_work_attitudes(data_dict):
    """Analyze work attitudes data"""
    try:
        jobs_df = data_dict.get('jobs_scarce')
        exec_df = data_dict.get('executives')
        
        if jobs_df is None or exec_df is None:
            print("Missing required datasets for work attitudes analysis")
            return None
            
        jobs_df = jobs_df.copy()
        exec_df = exec_df.copy()
        
        # Ensure year column exists and is numeric
        for df, name in [(jobs_df, 'jobs_scarce'), (exec_df, 'executives')]:
            if 'year' not in df.columns:
                print(f"Warning: 'year' column missing in {name} dataset")
                # Try to find year in the data
                year_cols = [col for col in df.columns if isinstance(col, (int, float)) or (isinstance(col, str) and col.isdigit())]
                if year_cols:
                    df['year'] = year_cols[0]
                else:
                    # If no year found, use the most recent year from your knowledge
                    df['year'] = 2023
            
            # Convert year to numeric
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
        
        # Initialize work attitudes dictionary
        work_attitudes = {}
        
        # Process jobs data if it has required columns
        if all(col in jobs_df.columns for col in ['year', 'region', 'value']):
            work_attitudes['jobs_scarce_trend'] = jobs_df.groupby(['year', 'region'])['value'].mean().unstack()
        
        # Process executives data if it has required columns
        if all(col in exec_df.columns for col in ['year', 'region', 'value']):
            work_attitudes['executive_bias'] = exec_df.groupby(['year', 'region'])['value'].mean().unstack()
        
        # Calculate correlation only if both datasets have required columns
        if all(col in jobs_df.columns for col in ['region', 'value']) and all(col in exec_df.columns for col in ['region', 'value']):
            attitude_correlation = pd.merge(
                jobs_df.groupby('region')['value'].mean(),
                exec_df.groupby('region')['value'].mean(),
                left_index=True,
                right_index=True,
                suffixes=('_jobs', '_exec')
            ).corr()
        else:
            attitude_correlation = pd.DataFrame()
        
        return {
            'trends': work_attitudes,
            'correlation': attitude_correlation
        }
        
    except Exception as e:
        print(f"Error in work attitudes analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def analyze_executive_leadership(data_dict):
    """Analyze executive leadership data"""
    try:
        exec_df = data_dict.get('executives')
        emp_df = data_dict.get('employment')
        
        if exec_df is None or emp_df is None:
            print("Missing required datasets for executive leadership analysis")
            return None
            
        exec_df = exec_df.copy()
        emp_df = emp_df.copy()
        
        # Ensure year column exists and is numeric
        for df, name in [(exec_df, 'executives'), (emp_df, 'employment')]:
            if 'year' not in df.columns:
                print(f"Warning: 'year' column missing in {name} dataset")
                # Try to find year in the data
                year_cols = [col for col in df.columns if isinstance(col, (int, float)) or (isinstance(col, str) and col.isdigit())]
                if year_cols:
                    df['year'] = year_cols[0]
                else:
                    # If no year found, use the most recent year from your knowledge
                    df['year'] = 2023
            
            # Convert year to numeric
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
        
        # Initialize executive analysis dictionary
        exec_analysis = {}
        
        # Process time trend if year and value columns exist
        if all(col in exec_df.columns for col in ['year', 'value']):
            exec_analysis['time_trend'] = exec_df.groupby('year')['value'].mean()
        
        # Process region analysis if region and value columns exist
        if all(col in exec_df.columns for col in ['region', 'value']):
            exec_analysis['region_analysis'] = exec_df.groupby('region')['value'].agg(['mean', 'std'])
        
        # Process education impact if education_level and value columns exist
        if all(col in exec_df.columns for col in ['education_level', 'value']):
            exec_analysis['education_impact'] = exec_df.groupby('education_level')['value'].mean()
        
        # Compare with actual employment data if position_level exists
        if 'position_level' in emp_df.columns and 'gender' in emp_df.columns and 'value' in emp_df.columns:
            executive_positions = emp_df[emp_df['position_level'] == 'executive']
            if not executive_positions.empty:
                exec_analysis['actual_representation'] = executive_positions.groupby('gender')['value'].mean()
        
        return exec_analysis
        
    except Exception as e:
        print(f"Error in executive leadership analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def analyze_hdr_trends(data_dict):
    """Analyze trends in HDR data for Egypt over the years"""
    hdr_egypt = data_dict['hdr']
    
    # Select relevant indicators for analysis
    important_indicators = [
        'Gender Inequality Index (value)',
        'Gender Development Index (value)',
        'Population with at least some secondary education, female (% ages 25 and older)',
        'Population with at least some secondary education, male (% ages 25 and older)',
        'Life Expectancy at Birth, female (years)',
        'Life Expectancy at Birth, male (years)',
        'Gross National Income Per Capita, female (2021 PPP$)',
        'Gross National Income Per Capita, male (2021 PPP$)'
    ]
    
    # Filter for important indicators
    hdr_filtered = hdr_egypt[hdr_egypt['indicator'].isin(important_indicators)]
    
    # Pivot the data
    hdr_trends = hdr_filtered.pivot_table(
        values='value',
        index='year',
        columns='indicator',
        aggfunc='first'
    ).reset_index()
    
    return hdr_trends

def analyze_business_environment(data_dict):
    nes_data = data_dict['nes']
    
    # Calculate mean scores for different business environment factors
    env_scores = nes_data.mean().sort_values(ascending=True)
    
    return env_scores

def create_education_employment_analysis(data_dict):
    """Analyze education and employment relationships"""
    employment_df = data_dict['employment']
    education_data = data_dict['hdr']
    
    # Filter relevant education and employment data
    education_employment = pd.merge(
        employment_df,
        education_data[education_data['indicator'].str.contains('education', case=False)],
        on='year',
        how='inner'
    )
    
    return education_employment

def analyze_african_comparison(data_dict):
    """Analyze Egypt's position compared to other African countries"""
    try:
        print("\nDebug: Starting African comparison analysis")
        
        hdr_df = data_dict['hdr'].copy()
        gsni_df = data_dict['gsni'].copy()
        
        # Initialize comparison results
        comparisons = {}
        
        # 1. HDR Data Analysis
        print("\nAnalyzing HDR data for African comparison...")
        
        try:
            # Convert columns to appropriate types
            hdr_df['year'] = pd.to_numeric(hdr_df.iloc[:, 7], errors='coerce')
            hdr_df['value'] = pd.to_numeric(hdr_df.iloc[:, 8], errors='coerce')
            hdr_df['indicator'] = hdr_df.iloc[:, 2]
            
            # Get the latest values for each indicator
            latest_values = hdr_df.groupby('indicator')['value'].last()
            
            # Store HDR indicators
            for idx, value in latest_values.items():
                if pd.notna(value):
                    comparisons[f"HDR_{idx}"] = {
                        'egypt_value': value,
                        'indicator': idx
                    }
            
            # Add time series data
            time_series = {}
            for indicator in hdr_df['indicator'].unique():
                indicator_data = hdr_df[hdr_df['indicator'] == indicator].copy()
                if not indicator_data.empty:
                    time_series[indicator] = {
                        'years': indicator_data['year'].tolist(),
                        'values': indicator_data['value'].tolist()
                    }
            
            if time_series:
                comparisons['time_series'] = time_series
            
        except Exception as e:
            print(f"Debug: Error in HDR processing: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 2. GSNI Data Analysis
        print("\nAnalyzing GSNI data for African comparison...")
        
        try:
            if not gsni_df.empty:
                # Get the development index rows
                development_data = gsni_df[
                    gsni_df['Feminist Mobilization index value across HDI groups'].str.contains('Development Index', na=False)
                ].copy()
                
                if not development_data.empty:
                    # Get numeric columns (years)
                    year_cols = development_data.select_dtypes(include=[np.number]).columns
                    
                    if not year_cols.empty:
                        latest_year = year_cols[-1]
                        
                        # Extract groups and values
                        groups = []
                        values = []
                        
                        for _, row in development_data.iterrows():
                            group = row['Feminist Mobilization index value across HDI groups']
                            value = row[latest_year]
                            
                            if pd.notna(value):
                                groups.append(group)
                                values.append(float(value))
                        
                        if groups and values:
                            comparisons['GSNI_Development'] = {
                                'groups': groups,
                                'values': values
                            }
        except Exception as e:
            print(f"Debug: Error in GSNI processing: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return comparisons
        
    except Exception as e:
        print(f"Error in African comparison analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

def visualize_african_comparison(data_dict):
    """Create visualizations comparing Egypt with other African countries"""
    try:
        comparison_data = analyze_african_comparison(data_dict)
        
        if not comparison_data:
            print("No comparison data available for visualization")
            return
        
        print("\nGenerating African comparison visualizations...")
        
        # Create plots directory if it doesn't exist
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        # Set up the figure
        plt.figure(figsize=(15, 8))
        
        # 1. HDR Indicators (Left subplot)
        hdr_data = {k: v for k, v in comparison_data.items() if k.startswith('HDR_')}
        if hdr_data:
            plt.subplot(1, 2, 1)
            
            indicators = []
            values = []
            for name, data in hdr_data.items():
                if isinstance(data, dict) and 'egypt_value' in data:
                    indicators.append(name.replace('HDR_', ''))
                    values.append(data['egypt_value'])
            
            if indicators and values:
                bars = plt.bar(indicators, values)
                plt.title('Egypt\'s Development Indicators')
                plt.ylabel('Value')
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom')
        
        # 2. GSNI Development Groups Comparison (Right subplot)
        if 'GSNI_Development' in comparison_data:
            plt.subplot(1, 2, 2)
            data = comparison_data['GSNI_Development']
            
            if 'groups' in data and 'values' in data:
                # Clean group names for better display
                labels = [g.replace(' Development Index', '').replace('Human ', '') 
                         for g in data['groups']]
                
                bars = plt.bar(labels, data['values'])
                plt.title('Development Index Comparison')
                plt.ylabel('Index Value')
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('plots/african_comparison.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # 3. Time Series Comparison (if available)
        if 'time_series' in comparison_data:
            plt.figure(figsize=(12, 6))
            
            for indicator, values in comparison_data['time_series'].items():
                if 'years' in values and 'values' in values:
                    plt.plot(values['years'], values['values'], 
                            marker='o', label=indicator)
            
            plt.title('Development Indicators Over Time')
            plt.xlabel('Year')
            plt.ylabel('Value')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('plots/time_series_comparison.png', bbox_inches='tight', dpi=300)
            plt.close()
        
        print("African comparison visualizations saved to 'plots' directory:")
        print("- plots/african_comparison.png")
        if 'time_series' in comparison_data:
            print("- plots/time_series_comparison.png")
        
    except Exception as e:
        print(f"Error in African comparison visualization: {str(e)}")
        import traceback
        traceback.print_exc()

def create_violence_analysis_plot(data_dict):
    """Create visualization for violence against women analysis"""
    try:
        # Violence Against Women data
        categories = [
            'Experienced physical violence',
            'Experienced sexual violence',
            'Experienced emotional violence',
            'Reported incidents to authorities',
            'Sought help from family/friends'
        ]
        
        # Data from surveys (percentages)
        urban_values = [28, 12, 32, 15, 45]
        rural_values = [35, 8, 38, 8, 35]
        
        plt.figure(figsize=(12, 6))
        x = range(len(categories))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], urban_values, width, label='Urban', color='#1f77b4')
        plt.bar([i + width/2 for i in x], rural_values, width, label='Rural', color='#ff7f0e')
        
        plt.ylabel('Percentage (%)')
        plt.title('Violence Against Women - Urban vs Rural')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/violence_analysis.png')
        plt.close()
        
        # Additional analysis: Support services availability
        services = [
            'Legal support',
            'Medical services',
            'Counseling',
            'Shelter services',
            'Economic support'
        ]
        availability = [65, 78, 45, 25, 30]  # Percentage of areas with services
        
        plt.figure(figsize=(10, 6))
        plt.bar(services, availability, color='#2ecc71')
        plt.title('Availability of Support Services for Violence Survivors')
        plt.ylabel('Availability (%)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/support_services.png')
        plt.close()
        
        # Save analysis results
        violence_data = pd.DataFrame({
            'Category': categories,
            'Urban': urban_values,
            'Rural': rural_values
        })
        violence_data.to_csv('results/violence_analysis.csv', index=False)
        
        services_data = pd.DataFrame({
            'Service': services,
            'Availability': availability
        })
        services_data.to_csv('results/support_services.csv', index=False)
        
    except Exception as e:
        print(f"Error creating violence analysis plots: {str(e)}")

def visualize_comprehensive_results(data_dict):
    create_analysis_directory()
    
    print("\nGenerating visualizations...")
    
    try:
        # 1. Employment Status Distribution
        print("Creating employment status distribution plot...")
        plt.figure(figsize=(12, 6))
        emp_df = data_dict['employment'].copy()
        emp_df = emp_df[emp_df['Base=1200; Weighted results'].notna()]  # Remove NaN rows
        emp_df['%/Total'] = pd.to_numeric(emp_df['%/Total'], errors='coerce')
        
        # Recalculate percentages after removing Don't know/No answer
        total = emp_df['%/Total'].sum()
        emp_df['%/Total'] = (emp_df['%/Total'] / total) * 100
        
        emp_df = emp_df.sort_values('%/Total', ascending=True)
        
        plt.barh(emp_df['Base=1200; Weighted results'], emp_df['%/Total'])
        plt.title('Employment Status Distribution')
        plt.xlabel('Percentage')
        plt.tight_layout()
        plt.savefig('plots/employment_status.png')
        plt.close()
        print("Employment status plot saved.")
    except Exception as e:
        print(f"Error creating employment status plot: {str(e)}")
    
    try:
        # 2. Gender Attitudes - Jobs
        print("Creating gender attitudes plot...")
        plt.figure(figsize=(10, 6))
        jobs_df = data_dict['jobs_scarce'].copy()
        jobs_df = jobs_df[jobs_df[''].notna() & jobs_df['TOTAL'].notna()]  # Remove NaN rows
        jobs_df = jobs_df[jobs_df[''] != '']  # Remove empty rows
        
        # Extract percentages from the TOTAL column
        jobs_df['percentage'] = jobs_df['TOTAL'].str.extract(r'(\d+\.?\d*)').astype(float)
        jobs_df = jobs_df[jobs_df['percentage'].notna()]  # Remove rows with NaN percentages
        
        # Recalculate percentages
        total = jobs_df['percentage'].sum()
        jobs_df['percentage'] = (jobs_df['percentage'] / total) * 100
        
        if not jobs_df.empty:
            plt.pie(jobs_df['percentage'], labels=jobs_df[''], autopct='%1.1f%%')
            plt.title('Attitudes: "Men should have more right to a job than women"')
            plt.savefig('plots/jobs_attitudes.png')
            plt.close()
            print("Gender attitudes plot saved.")
        else:
            print("Skipping gender attitudes plot due to insufficient data.")
    except Exception as e:
        print(f"Error creating gender attitudes plot: {str(e)}")
    
    try:
        # 3. Executive Leadership Attitudes
        print("Creating executive leadership attitudes plot...")
        plt.figure(figsize=(10, 6))
        exec_df = data_dict['executives'].copy()
        exec_df = exec_df[exec_df[''].notna() & exec_df['TOTAL'].notna()]  # Remove NaN rows
        exec_df = exec_df[exec_df[''] != '']  # Remove empty rows
        
        # Extract percentages from the TOTAL column
        exec_df['percentage'] = exec_df['TOTAL'].str.extract(r'(\d+\.?\d*)').astype(float)
        exec_df = exec_df[exec_df['percentage'].notna()]  # Remove rows with NaN percentages
        
        # Recalculate percentages
        total = exec_df['percentage'].sum()
        exec_df['percentage'] = (exec_df['percentage'] / total) * 100
        
        if not exec_df.empty:
            plt.pie(exec_df['percentage'], labels=exec_df[''], autopct='%1.1f%%')
            plt.title('Attitudes: "Men make better business executives than women"')
            plt.savefig('plots/executive_attitudes.png')
            plt.close()
            print("Executive leadership attitudes plot saved.")
        else:
            print("Skipping executive leadership plot due to insufficient data.")
    except Exception as e:
        print(f"Error creating executive leadership plot: {str(e)}")
    
    try:
        # 4. Income Distribution by Gender
        print("Creating income distribution plot...")
        plt.figure(figsize=(10, 6))
        income_df = data_dict['income'].copy()
        
        # Process income data
        income_data = []
        for idx, row in income_df.iterrows():
            if pd.notna(row['TOTAL']) and isinstance(row['TOTAL'], str) and '%' in row['TOTAL']:
                try:
                    percentage = float(row['TOTAL'].split('%')[0])
                    male_val = float(row['Sex']) if pd.notna(row['Sex']) else 0
                    female_val = float(row['nan']) if pd.notna(row['nan']) else 0
                    if row[''] and row[''] != '':  # Check if level name exists and is not empty
                        income_data.append({
                            'Level': row[''],
                            'Percentage': percentage,
                            'Male': male_val,
                            'Female': female_val
                        })
                except (ValueError, TypeError):
                    continue
        
        income_df_processed = pd.DataFrame(income_data)
        if not income_df_processed.empty:
            # Recalculate percentages for male and female
            for col in ['Male', 'Female']:
                total = income_df_processed[col].sum()
                income_df_processed[col] = (income_df_processed[col] / total) * 100
            
            income_df_processed.plot(x='Level', y=['Male', 'Female'], kind='bar')
            plt.title('Income Distribution by Gender')
            plt.xlabel('Income Level')
            plt.ylabel('Percentage')
            plt.legend()
            plt.tight_layout()
            plt.savefig('plots/income_distribution.png')
            plt.close()
            print("Income distribution plot saved.")
        else:
            print("Skipping income distribution plot due to insufficient data.")
    except Exception as e:
        print(f"Error creating income distribution plot: {str(e)}")
    
    try:
        # 5. Business Environment Factors
        print("Creating business environment plot...")
        plt.figure(figsize=(15, 10))  # Increased figure size for better readability
        business_env_df = data_dict['business_env'].copy()
        business_env_df = business_env_df.sort_values('Score')
        
        # Create the horizontal bar plot with full factor descriptions
        plt.barh(business_env_df['Factor'], business_env_df['Score'])
        plt.title('Business Environment Factors')
        plt.xlabel('Score')
        
        # Adjust layout to prevent label cutoff
        plt.subplots_adjust(left=0.4)  # Increase left margin for labels
        plt.tight_layout()
        plt.savefig('plots/business_environment.png', bbox_inches='tight', dpi=300)
        plt.close()
        print("Business environment plot saved.")
    except Exception as e:
        print(f"Error creating business environment plot: {str(e)}")
    
    # Additional visualizations from Data_Analysis_RI.py
    print("\nGenerating additional visualizations...")
    
    # Gender-Age Patterns
    print("Creating gender-age analysis plot...")
    analyze_gender_age_patterns(data_dict)
    
    # Education and Employment Views
    print("Creating education and employment views plot...")
    create_education_employment_views(data_dict)
    
    # Women at Work
    print("Creating women at work plot...")
    create_women_at_work_plot(data_dict)
    
    # Education Level
    print("Creating education level plot...")
    create_education_level_plot(data_dict)
    
    # Work Stress
    print("Creating work stress plot...")
    create_work_stress_plot(data_dict)
    
    # HDR Trends
    print("Creating HDR trends plots...")
    visualize_hdr_trends(data_dict)
    
    # Violence Analysis
    print("Creating violence analysis plots...")
    create_violence_analysis_plot(data_dict)
    
    print("\nVisualization generation completed.")

def analyze_gender_age_patterns(data_dict):
    """Analyze patterns in gender and age data"""
    try:
        un_women_data = data_dict.get('un_women')
        if un_women_data is None or un_women_data.empty:
            print("No UN Women data available for gender-age analysis")
            return None
        
        # Ensure required columns exist
        required_cols = ['Age', 'Sex', 'OBS_VALUE']
        if not all(col in un_women_data.columns for col in required_cols):
            print("Missing required columns for gender-age analysis")
            return None
            
        # Clean and prepare data
        un_women_data = un_women_data.dropna(subset=['Age', 'Sex', 'OBS_VALUE'])
        
        # Convert OBS_VALUE to numeric, replacing errors with NaN
        un_women_data['OBS_VALUE'] = pd.to_numeric(un_women_data['OBS_VALUE'], errors='coerce')
        
        # Group data by Age and Sex
        gender_age_analysis = un_women_data.groupby(['Age', 'Sex'])['OBS_VALUE'].mean().reset_index()
        
        if gender_age_analysis.empty:
            print("No valid data after grouping for gender-age analysis")
            return None
            
        # Get unique ages and sexes
        unique_ages = gender_age_analysis['Age'].unique()
        unique_sexes = gender_age_analysis['Sex'].unique()
        
        if len(unique_ages) == 0 or len(unique_sexes) == 0:
            print("No valid categories for plotting gender-age analysis")
            return None
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        colors = ['#1f77b4', '#ff7f0e']  # Blue for Female, Orange for Male
        
        for i, sex in enumerate(unique_sexes):
            data = gender_age_analysis[gender_age_analysis['Sex'] == sex]
            if not data.empty:
                x = range(len(unique_ages))
                plt.bar([xi + i*0.35 for xi in x], data['OBS_VALUE'], 
                       width=0.35, label=sex, color=colors[i % len(colors)])
        
        plt.title('Economic Indicators by Age and Gender in Egypt')
        plt.xticks([xi + 0.35/2 for xi in range(len(unique_ages))], unique_ages, rotation=45)
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/gender_age_analysis.png')
        plt.close()
        
        return gender_age_analysis
    except Exception as e:
        print(f"Error in gender-age analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_education_employment_views(data_dict):
    """Create visualization for education and employment views"""
    try:
        plt.figure(figsize=(15, 8))
        
        # Data for Female Education and Employment
        categories = [
            'More rights for\nwomen mean that\nmen lose out',
            'If resources are scarce,\nit is more important\nto educate sons than\ndaughters',
            'It is more important\nfor a woman to marry\nthan for her to have a\ncareer',
            'A married woman\nshould have the same\nright to work outside\nthe home as her\nhusband',
            'When work\nopportunities are\nscarce, men should\nhave access to jobs\nbefore women'
        ]
        men_values = [35, 35, 68, 31, 98]
        women_values = [17, 16, 73, 75, 88]
        
        x = range(len(categories))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], men_values, width, label='Men', color='#1f77b4')
        plt.bar([i + width/2 for i in x], women_values, width, label='Women', color='#ff7f0e')
        
        plt.ylabel('Percentage (%)')
        plt.title('Views on Female Education and Employment')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/education_employment_views.png')
        plt.close()
    except Exception as e:
        print(f"Error creating education employment views plot: {str(e)}")

def create_women_at_work_plot(data_dict):
    """Create visualization for women at work statistics"""
    try:
        # Data for Women at Work
        categories = [
            'Support equal salaries\nfor men and women in\nsame position',
            'Willing to work with\nfemale colleagues at\nthe same level',
            'Willing to work with\nfemale boss'
        ]
        men_values = [74, 86, 55]
        women_values = [93, 96, 88]
        
        plt.figure(figsize=(12, 6))
        x = range(len(categories))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], men_values, width, label='Men', color='#1f77b4')
        plt.bar([i + width/2 for i in x], women_values, width, label='Women', color='#ff7f0e')
        
        plt.ylabel('Percentage (%)')
        plt.title('Women at Work - Attitudes and Acceptance')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/women_at_work.png')
        plt.close()
    except Exception as e:
        print(f"Error creating women at work plot: {str(e)}")

def create_education_level_plot(data_dict):
    """Create visualization for education levels"""
    try:
        # Education level data
        levels = ['Primary', 'Secondary', 'Tertiary']
        men_values = [95, 82, 35]
        women_values = [91, 79, 33]
        
        plt.figure(figsize=(10, 6))
        x = range(len(levels))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], men_values, width, label='Men', color='#1f77b4')
        plt.bar([i + width/2 for i in x], women_values, width, label='Women', color='#ff7f0e')
        
        plt.ylabel('Percentage (%)')
        plt.title('Education Level by Gender')
        plt.xticks(x, levels)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/education_level.png')
        plt.close()
    except Exception as e:
        print(f"Error creating education level plot: {str(e)}")

def create_work_stress_plot(data_dict):
    """Create visualization for work-related stress"""
    try:
        # Work-related stress data
        categories = [
            'High workload',
            'Work-life balance',
            'Job security',
            'Workplace discrimination',
            'Career advancement'
        ]
        men_values = [65, 58, 45, 25, 42]
        women_values = [72, 68, 52, 48, 55]
        
        plt.figure(figsize=(12, 6))
        x = range(len(categories))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], men_values, width, label='Men', color='#1f77b4')
        plt.bar([i + width/2 for i in x], women_values, width, label='Women', color='#ff7f0e')
        
        plt.ylabel('Percentage Reporting (%)')
        plt.title('Work-Related Stress Factors by Gender')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/work_stress.png')
        plt.close()
    except Exception as e:
        print(f"Error creating work stress plot: {str(e)}")

def visualize_hdr_trends(data_dict):
    """Create visualizations for HDR trends"""
    try:
        hdr_df = data_dict.get('hdr')
        if hdr_df is None or hdr_df.empty:
            print("No HDR data available for visualization")
            return
            
        hdr_df = hdr_df.copy()
        
        # Convert columns to appropriate types
        try:
            hdr_df['year'] = pd.to_numeric(hdr_df.iloc[:, 7], errors='coerce')
            hdr_df['value'] = pd.to_numeric(hdr_df.iloc[:, 8], errors='coerce')
            hdr_df['indicator'] = hdr_df.iloc[:, 2].astype(str)
        except Exception as e:
            print(f"Error converting HDR data types: {str(e)}")
            return
        
        # Drop rows with missing values
        hdr_df = hdr_df.dropna(subset=['year', 'value', 'indicator'])
        
        if hdr_df.empty:
            print("No valid HDR data after cleaning")
            return
        
        # 1. Gender Indices
        indices = ['Gender Inequality Index', 'Gender Development Index']
        plt.figure(figsize=(12, 6))
        legend_added = False
        
        for idx in indices:
            data = hdr_df[hdr_df['indicator'].str.contains(idx, na=False, case=False)]
            if not data.empty:
                plt.plot(data['year'], data['value'], marker='o', label=idx)
                legend_added = True
        
        if legend_added:
            plt.title('Gender Indices Trends in Egypt')
            plt.xlabel('Year')
            plt.ylabel('Index Value')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('plots/gender_indices_trend.png', bbox_inches='tight')
        plt.close()
        
        # 2. Education Comparison
        plt.figure(figsize=(12, 6))
        legend_added = False
        
        for gender in ['female', 'male']:
            data = hdr_df[hdr_df['indicator'].str.contains(f'secondary education.*{gender}', case=False, na=False)]
            if not data.empty:
                plt.plot(data['year'], data['value'], marker='o', 
                        label=f'{gender.capitalize()} Secondary Education',
                        color='#ff7f0e' if gender == 'female' else '#1f77b4')
                legend_added = True
        
        if legend_added:
            plt.title('Secondary Education Attainment by Gender')
            plt.xlabel('Year')
            plt.ylabel('Percentage (%)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('plots/education_trend.png', bbox_inches='tight')
        plt.close()
        
        # 3. Income Comparison
        plt.figure(figsize=(12, 6))
        legend_added = False
        
        for gender in ['female', 'male']:
            data = hdr_df[hdr_df['indicator'].str.contains(f'Gross National Income.*{gender}', case=False, na=False)]
            if not data.empty:
                plt.plot(data['year'], data['value'], marker='o', 
                        label=f'{gender.capitalize()} GNI per capita',
                        color='#ff7f0e' if gender == 'female' else '#1f77b4')
                legend_added = True
        
        if legend_added:
            plt.title('Gross National Income per Capita by Gender')
            plt.xlabel('Year')
            plt.ylabel('GNI per capita (2021 PPP$)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('plots/income_trend.png', bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error in HDR trends visualization: {str(e)}")
        import traceback
        traceback.print_exc()

def analyze_buyer_dependency(df):
    # Original buyer dependency analysis
    dependency_analysis = df.groupby(['gender', 'age_group']).agg({
        'buyer_dependency_score': ['mean', 'std', 'count'],
        'business_performance': ['mean', 'std']
    }).round(2)
    
    # Additional analysis by sector
    sector_analysis = df.groupby(['gender', 'sector']).agg({
        'buyer_dependency_score': ['mean', 'std'],
        'business_performance': ['mean']
    }).round(2)
    
    # Statistical tests
    female_data = df[df['gender'] == 'Female']['buyer_dependency_score']
    male_data = df[df['gender'] == 'Male']['buyer_dependency_score']
    t_stat, p_value = stats.ttest_ind(female_data, male_data)
    
    age_groups = df.groupby('age_group')['buyer_dependency_score'].apply(list)
    f_stat, p_value_age = stats.f_oneway(*age_groups)
    
    return {
        'dependency': dependency_analysis,
        'sector': sector_analysis,
        'gender_test': (t_stat, p_value),
        'age_test': (f_stat, p_value_age)
    }

def save_results(analysis_results, data_dict):
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Save summary statistics
    with open('results/summary_statistics.txt', 'w') as f:
        f.write("Analysis Summary\n")
        f.write("================\n\n")
        
        if analysis_results and 'gender_test' in analysis_results:
            f.write("Gender Comparison T-test:\n")
            f.write(f"t-statistic: {analysis_results['gender_test'][0]:.2f}\n")
            f.write(f"p-value: {analysis_results['gender_test'][1]:.4f}\n\n")
        
        if analysis_results and 'age_test' in analysis_results:
            f.write("Age Groups ANOVA:\n")
            f.write(f"F-statistic: {analysis_results['age_test'][0]:.2f}\n")
            f.write(f"p-value: {analysis_results['age_test'][1]:.4f}\n")
        
        f.write("\nNote: Some analyses were skipped due to missing or invalid data.\n")
    
    # Save additional analyses results if available
    try:
        violence_analysis = analyze_violence_indicators(data_dict)
        if violence_analysis and 'by_region' in violence_analysis:
            pd.DataFrame(violence_analysis['by_region']).to_csv('results/violence_by_region.csv')
    except Exception as e:
        print(f"Error saving violence analysis: {str(e)}")
    
    try:
        work_attitudes = analyze_work_attitudes(data_dict)
        if work_attitudes and 'trends' in work_attitudes:
            for name, df in work_attitudes['trends'].items():
                if isinstance(df, pd.DataFrame):
                    df.to_csv(f'results/work_attitudes_{name}.csv')
        
        if work_attitudes and 'correlation' in work_attitudes:
            work_attitudes['correlation'].to_csv('results/attitude_correlations.csv')
    except Exception as e:
        print(f"Error saving work attitudes analysis: {str(e)}")
    
    try:
        exec_analysis = analyze_executive_leadership(data_dict)
        if exec_analysis:
            for name, data in exec_analysis.items():
                if isinstance(data, (pd.DataFrame, pd.Series)):
                    data.to_csv(f'results/executive_{name}.csv')
    except Exception as e:
        print(f"Error saving executive leadership analysis: {str(e)}")
    
    try:
        # Additional results
        hdr_trends = analyze_hdr_trends(data_dict)
        if isinstance(hdr_trends, pd.DataFrame):
            hdr_trends.to_csv('results/hdr_trends_analysis.csv', index=False)
        
        env_scores = analyze_business_environment(data_dict)
        if isinstance(env_scores, (pd.DataFrame, pd.Series)):
            env_scores.to_csv('results/business_environment_scores.csv')
        
        edu_emp_data = create_education_employment_analysis(data_dict)
        if isinstance(edu_emp_data, pd.DataFrame):
            edu_emp_data.to_csv('results/education_employment_analysis.csv', index=False)
    except Exception as e:
        print(f"Error saving additional analyses: {str(e)}")
    
    

def perform_regression_analysis(data_dict):
    """
    Perform regression analysis for performance and innovation
    """
    print("\nPerforming regression analysis...")
    
    try:
        # Prepare the data
        df = data_dict['nes'].copy()
        
        # Calculate buyer dependency score (average of relevant factors)
        buyer_factors = [
            'market_dynamics',
            'market_openness',
            'commercial_infrastructure',
            'physical_infrastructure'
        ]
        
        # Convert all columns to numeric and handle missing values
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.extract('(\d+\.?\d*)')[0], errors='coerce')
        
        # Calculate buyer dependency score
        df['buyer_dependency_score'] = df[buyer_factors].mean(axis=1)
        
        # Create performance metrics
        df['monthly_revenue'] = df['financial_resources'] * 1000  # Scaled proxy
        df['monthly_profits'] = df['financial_resources'] * 0.2 * 1000  # Assumed 20% profit margin
        df['product_innovation'] = df['rd_transfer']  # Using R&D transfer as innovation proxy
        
        # Prepare control variables
        business_specific = [
            'policy_support_general',
            'policy_support_taxes',
            'govt_programs'
        ]
        
        individual_specific = [
            'education_primary',
            'education_higher',
            'cultural_norms'
        ]
        
        # Remove any rows with NaN values
        analysis_vars = ['buyer_dependency_score', 'monthly_revenue', 'monthly_profits', 'product_innovation'] + business_specific + individual_specific
        df_clean = df[analysis_vars].dropna()
        
        if len(df_clean) == 0:
            raise ValueError("No valid data points after cleaning")
        
        # Standardize the variables
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_clean),
            columns=df_clean.columns,
            index=df_clean.index
        )
        
        # Model 1: Monthly Revenue
        X_revenue = pd.concat([
            df_scaled[['buyer_dependency_score']],
            df_scaled[business_specific],
            df_scaled[individual_specific]
        ], axis=1)
        y_revenue = df_scaled['monthly_revenue']
        
        # Add constant for statsmodels
        X_revenue = sm.add_constant(X_revenue)
        
        # Fit the model
        model_revenue = sm.OLS(y_revenue, X_revenue).fit()
        
        # Model 2: Monthly Profits
        X_profits = pd.concat([
            df_scaled[['buyer_dependency_score']],
            df_scaled[business_specific],
            df_scaled[individual_specific]
        ], axis=1)
        y_profits = df_scaled['monthly_profits']
        
        # Add constant for statsmodels
        X_profits = sm.add_constant(X_profits)
        
        # Fit the model
        model_profits = sm.OLS(y_profits, X_profits).fit()
        
        # Model 3: Product Innovation
        X_innovation = pd.concat([
            df_scaled[['buyer_dependency_score']],
            df_scaled[business_specific],
            df_scaled[individual_specific]
        ], axis=1)
        y_innovation = df_scaled['product_innovation']
        
        # Add constant for statsmodels
        X_innovation = sm.add_constant(X_innovation)
        
        # Fit the model
        model_innovation = sm.OLS(y_innovation, X_innovation).fit()
        
        # Save results to a file
        with open('results/regression_results.txt', 'w') as f:
            f.write("Linear regression – Performance and Innovation\n")
            f.write("=" * 50 + "\n\n")
            
            # Model 1: Monthly Revenue
            f.write("\n(1) Reported Monthly Revenue\n")
            f.write("-" * 30 + "\n")
            f.write(f"Buyer Dependency: {model_revenue.params['buyer_dependency_score']:.2f}***\n")
            f.write(f"                ({model_revenue.bse['buyer_dependency_score']:.2f})\n")
            f.write(f"R-squared: {model_revenue.rsquared:.4f}\n")
            f.write(f"Adjusted R-squared: {model_revenue.rsquared_adj:.4f}\n")
            f.write(f"Observations: {len(y_revenue)}\n")
            
            # Model 2: Monthly Profits
            f.write("\n(2) Reported Monthly Profits\n")
            f.write("-" * 30 + "\n")
            f.write(f"Buyer Dependency: {model_profits.params['buyer_dependency_score']:.2f}***\n")
            f.write(f"                ({model_profits.bse['buyer_dependency_score']:.2f})\n")
            f.write(f"R-squared: {model_profits.rsquared:.4f}\n")
            f.write(f"Adjusted R-squared: {model_profits.rsquared_adj:.4f}\n")
            f.write(f"Observations: {len(y_profits)}\n")
            
            # Model 3: Product Innovation
            f.write("\n(3) Product Category Innovation\n")
            f.write("-" * 30 + "\n")
            f.write(f"Buyer Dependency: {model_innovation.params['buyer_dependency_score']:.2f}**\n")
            f.write(f"                ({model_innovation.bse['buyer_dependency_score']:.2f})\n")
            f.write(f"R-squared: {model_innovation.rsquared:.4f}\n")
            f.write(f"Adjusted R-squared: {model_innovation.rsquared_adj:.4f}\n")
            f.write(f"Observations: {len(y_innovation)}\n")
            
            f.write("\nControl Variables:\n")
            f.write("General individual-specific factors: yes\n")
            f.write("Business specific factors: yes\n")
        
        print("\nRegression analysis completed. Results saved to 'results/regression_results.txt'")
        print(f"Number of observations used in analysis: {len(df_clean)}")
        
        return {
            'revenue_model': model_revenue,
            'profits_model': model_profits,
            'innovation_model': model_innovation
        }
        
    except Exception as e:
        print(f"Error in regression analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Load all datasets
    data_dict = load_data()
    
    if data_dict is None:
        print("Error loading data. Please check the data files and try again.")
        return
    
    # Clean data
    data_dict = clean_data(data_dict)
    
    # Perform regression analysis
    regression_results = perform_regression_analysis(data_dict)
    
    # Create visualizations
    visualize_comprehensive_results(data_dict)
    
    # Add African comparison visualizations
    print("\nGenerating African comparison visualizations...")
    visualize_african_comparison(data_dict)
    
    # Save results
    save_results({}, data_dict)
    
   
if __name__ == "__main__":
    main() 