import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

# Define the Streamlit application
def main():
    # File uploader for the master dataset
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    
    if uploaded_file is not None:
        # Load the Excel file
        df = pd.read_excel(uploaded_file)

        # Sidebar: Date range filter
        st.sidebar.header("Filter Options")
        start_date = st.sidebar.date_input("From Date", value=pd.to_datetime('2024-01-01'))
        end_date = st.sidebar.date_input("Until Date", value=pd.to_datetime('2024-12-31'))

        # Filter DataFrame by date range
        df_filtered = filter_by_date(df, 'demo_screening_date', start_date, end_date)

        # Sidebar: Visualization options
        st.sidebar.header("Visualization Options")
        visualization = st.sidebar.selectbox("Choose a visualization", 
                                             ['Eligibility Status', 'Demographics', 'Overall Health', 
                                              'MoCA Score Distribution', 'Physical Measurements', 
                                              'Participant Experience'])

        # Plot based on the user's selection
        if visualization == 'Eligibility Status':
            plot_eligibility_status(final_df)
        elif visualization == 'Demographics':
            plot_demographics(final_df)
        elif visualization == 'Overall Health':
            plot_overall_health(final_df)
        elif visualization == 'MoCA Score Distribution':
            plot_moca_score(final_df)
        elif visualization == 'Physical Measurements':
            plot_physical_measurements(final_df)
        elif visualization == 'Participant Experience':
            plot_participant_experience(final_df)

def filter_by_date(df, date_column, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    mask = (pd.to_datetime(df[date_column]) >= start_date) & (pd.to_datetime(df[date_column]) < end_date)
    return df[mask]

def plot_eligibility_status(df):
    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    
    df['eligibility_status'] = df['exc_eligible_2']
    sns.countplot(x='eligibility_status', data=df, palette='pastel', ax=ax)
    ax.set_title('Eligibility Status')
    ax.set_xlabel('')
    add_value_labels(ax, df)
    adjust_yaxis(ax)
    
    st.pyplot(fig)

def plot_demographics(df):
    # Define the order for education levels
    education_order = [
        'High school graduate', '12th grade or less', 'Some college',
        'Associate\'s Degree', 'Bachelor\'s Degree', 'Master\'s Degree', 'Doctorate or higher'
    ]
    race_order = [
        'White or Caucasian', 'Black or African-American', 
        'Asian or Pacific Islander', 'Hispanic or Latino',
        'Multiracial or Biracial', 'Not listed here', 'Native American or Alaskan American'
    ]

    # Convert the 'demo_highest_edu' and 'demo_race' columns to categorical types with the specified order
    df['demo_highest_edu'] = pd.Categorical(
        df['demo_highest_edu'], categories=education_order, ordered=True
    )
    df['demo_race'] = pd.Categorical(
        df['demo_race'], categories=race_order, ordered=True
    )

    demographic_cols = ['demo_gender', 'demo_race', 'demo_highest_edu', 'demo_current_employ', 'demo_current_marital']
    for col in demographic_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        if col == 'demo_highest_edu':
            valid_categories = df[col].dropna().value_counts().index.tolist()
            valid_categories = [cat for cat in education_order if cat in valid_categories]
            df_filtered = df[df[col].isin(valid_categories)]
            plot = sns.countplot(x=col, data=df_filtered, palette='pastel', order=valid_categories, ax=ax)
        elif col == 'demo_race':
            valid_categories = df[col].dropna().value_counts().index.tolist()
            valid_categories = [cat for cat in race_order if cat in valid_categories]
            df_filtered = df[df[col].isin(valid_categories)]
            plot = sns.countplot(x=col, data=df_filtered, palette='pastel', order=valid_categories, ax=ax)
        else:
            plot = sns.countplot(x=col, data=df, palette='pastel', ax=ax)
        add_value_labels(ax, df[col].dropna())
        adjust_yaxis(ax)
        plt.title(f"{col.replace('demo_', '').replace('_', ' ').title()} Distribution")
        plt.xlabel('')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Special additional graph for gender distribution within each race category
        if col == 'demo_race':
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x='demo_race', hue='demo_gender', data=df_filtered, palette='pastel', order=valid_categories, ax=ax)
            plt.title('Gender Distribution Within Each Race Category')
            plt.xlabel('Race')
            plt.ylabel('Count')
            plt.legend(title='Gender')
            plt.xticks(rotation=45)
            add_value_labels(ax, df[['demo_race', 'demo_gender']].dropna())
            adjust_yaxis(ax)
            st.pyplot(fig)
    
    # Age histogram with mean and standard deviation
    mean_age = df['demo_age'].mean()
    std_dev_age = df['demo_age'].std()

    fig, ax = plt.subplots(figsize=(10, 6))
    age_bins = range(50, 91, 5)
    sns.histplot(df['demo_age'].dropna(), bins=age_bins, kde=False, color='skyblue', ax=ax)
    add_value_labels(ax, df['demo_age'].dropna())
    adjust_yaxis(ax)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.xticks(age_bins)
    st.pyplot(fig)

    st.write(f"**Mean Age:** {mean_age:.2f}")
    st.write(f"**Standard Deviation:** {std_dev_age:.2f}")


def plot_overall_health(df):
    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)

    sns.boxplot(y='rating_overall_health', data=df, color='#FFA07A', orient='v', showmeans=True, ax=ax)
    ax.set_title('Overall Health Rating Distribution')

    mean_rating = df['rating_overall_health'].mean()
    median_rating = df['rating_overall_health'].median()
    std_rating = df['rating_overall_health'].std()

    stats_text = f'Mean: {mean_rating:.2f}\nMedian: {median_rating}\nStd: {std_rating:.2f}'
    ax.text(1.05, 0.95, stats_text, transform=ax.transAxes, ha='left', va='top')

    st.pyplot(fig)

def plot_moca_score(df):
    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)

    data_col = 'total_moca_2'
    color = '#20B2AA'

    sns.boxplot(y=data_col, data=df, color=color, orient='v', showmeans=True, ax=ax)
    ax.set_title('Total MoCA Score Distribution')
    ax.set_ylabel('Total MoCA Score')

    mean = df[data_col].mean()
    median = df[data_col].median()
    std = df[data_col].std()

    ax.text(1.05, 0.95, f'Mean: {mean:.2f}\nMedian: {median}\nStd: {std:.2f}', 
            transform=ax.transAxes, ha='left', va='top', fontsize=9)

    st.pyplot(fig)

def plot_physical_measurements(df):
    # Statistical Description of Physical Measurements
    physical_measures_cols = ['phy_height_inch', 'phy_weight_lb', 'phy_bmi', 'phy_arm', 'phy_sternal']
    physical_measures_names = ['Height (inches)', 'Weight (lbs)', 'BMI', 'Arm circumference (cm)', 'Sternal length (inch)']

    physical_desc = df[physical_measures_cols].describe().transpose()
    physical_desc.index = physical_measures_names
    formatted_physical_desc = physical_desc.applymap(lambda x: f'{x:.2f}')

    st.subheader("Descriptive Statistics of Physical Measurements")
    st.table(formatted_physical_desc)

    # Adding histogram for phy_skin
    skin_types = ['Type I (Pale white)', 'Type II (Fair)', 'Type III (Medium)', 'Type IV (Olive)', 'Type V (Brown)', 'Type VI (Very Dark)']
    skin_counts = df['phy_skin'].value_counts().reindex(skin_types, fill_value=0)
    colors = ['#FFDDC1', '#FCCB95', '#F8B888', '#D8A378', '#C68642', '#8C6239']

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=skin_types, y=skin_counts.values, palette=colors, ax=ax)

    total = len(df['phy_skin'])
    for rect in ax.patches:
        height = rect.get_height()
        percentage = (height / total) * 100
        ax.text(rect.get_x() + rect.get_width() / 2., height + 0.1, 
                f'{int(height)}\n({percentage:.1f}%)', ha='center', va='bottom')

    ax.set_xticklabels(skin_types, rotation=45)
    ax.set_ylabel('Number of Participants')
    ax.set_title('Distribution of Skin Types')

    highest_count = max([p.get_height() for p in ax.patches])
    upper_limit = highest_count + (0.2 * highest_count)
    ax.set_ylim(0, upper_limit)
    ax.set_yticks(np.arange(0, upper_limit, step=max(1, highest_count // 5)))

    st.pyplot(fig)
def plot_participant_experience(df):
    # Calculate the averages of both 'pe_easy' and 'pe_valuable'
    avg_pe_easy = df['pe_easy'].mean()
    avg_pe_valuable = df['pe_valuable'].mean()

    # Display the results
    st.write(f"**Average ease of participation:** {avg_pe_easy:.2f}")
    st.write(f"**Average perceived value:** {avg_pe_valuable:.2f}")

    # Plot distribution of 'pe_easy'
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['pe_easy'], bins=10, kde=False, color='lightgreen', ax=ax)  # kde=False removes the KDE line
    ax.set_title('Distribution of Ease of Use', fontsize=14)
    ax.set_xlabel('Ease of Use', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    add_value_labels(ax, df['pe_easy'])
    adjust_yaxis(ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True)  # Enable the grid lines
    st.pyplot(fig)

    # Plot distribution of 'pe_valuable'
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['pe_valuable'], bins=10, kde=False, color='lightgreen', ax=ax)  # kde=False removes the KDE line
    ax.set_title('Distribution of Perceived Value', fontsize=14)
    ax.set_xlabel('Perceived Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    add_value_labels(ax, df['pe_valuable'])
    adjust_yaxis(ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True)  # Enable the grid lines
    st.pyplot(fig)


def calculate_failing_rate_and_p_value(df, column):
    total_counts = df[column].value_counts(dropna=False)
    out_of_spec_counts = df[df['good_readings'].isna()][column].value_counts(dropna=False)

    failing_rate = (out_of_spec_counts / total_counts) * 100
    failing_rate = failing_rate.fillna(0).round(1).astype(str) + '%'

    contingency_table = pd.crosstab(df[column], df['good_readings'].isna())
    chi2, p, _, _ = stats.chi2_contingency(contingency_table)

    return failing_rate, p

def add_value_labels(ax, data):
    total = len(data)
    for rect in ax.patches:
        height = rect.get_height()
        percentage = (height / total) * 100
        ax.text(rect.get_x() + rect.get_width() / 2., height + 0.1, 
                f'{int(height)}\n({percentage:.1f}%)', ha='center', va='bottom')

def adjust_yaxis(ax):
    if len(ax.patches) > 0:
        highest_count = max([p.get_height() for p in ax.patches])
        upper_limit = highest_count + (0.2 * highest_count)
        ax.set_ylim(0, upper_limit)
        step = max(1, int(highest_count / 5))
        ax.set_yticks(np.arange(0, upper_limit, step=step))

if __name__ == "__main__":
    main()
