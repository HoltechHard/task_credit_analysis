"""
preprocessing.py
================
Enhanced data preprocessing module for Credit Score Classification project.
Provides comprehensive data cleaning, EDA visualizations, and feature engineering tools.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from scipy import stats

# Set global style for consistent visualizations
sbn.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})


class DataPreprocessing:
    """Comprehensive data preprocessing and exploratory visualization class."""

    # =========================================================================
    # 1. METADATA
    # =========================================================================

    def get_metadata(self, data):
        """Return column names, numerical columns, and categorical columns."""
        metadata = data.columns
        numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
        categorical_cols = data.select_dtypes(include=["object", "bool"]).columns.tolist()
        return metadata, numerical_cols, categorical_cols

    def dataset_summary(self, data):
        """Print a structured summary of the dataset: shape, dtypes, missing values, duplicates."""
        print("=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"  Rows: {data.shape[0]:,}  |  Columns: {data.shape[1]}")
        print(f"  Duplicates: {data.duplicated().sum():,}")
        print(f"  Total missing values: {data.isnull().sum().sum():,}")
        print(f"  Memory usage: {data.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        print("-" * 60)
        print("\nData types:")
        for dtype, count in data.dtypes.value_counts().items():
            print(f"  {str(dtype):15s} : {count}")
        print("\nMissing values per column (> 0):")
        missing = data.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing) == 0:
            print("  None")
        else:
            for col, val in missing.items():
                pct = val / len(data) * 100
                print(f"  {col:35s} : {val:>6,}  ({pct:.1f}%)")
        print("=" * 60)

    # =========================================================================
    # 2. MISSING DATA VISUALIZATION
    # =========================================================================

    def filter_missing(self, data, title="Missing Data Analysis"):
        """Visualize missing data as a stacked bar chart."""
        missing_pct = (data.isnull().sum() / len(data) * 100).sort_values(ascending=True)
        missing_pct = missing_pct[missing_pct > 0]

        if missing_pct.empty:
            print("No missing values found in the dataset.")
            return

        plt.figure(figsize=(10, max(4, len(missing_pct) * 0.35)))
        colors = sbn.color_palette("YlOrRd", n_colors=len(missing_pct))
        sbn.barplot(x=missing_pct.values, y=missing_pct.index, palette=colors)

        for i, (val, col) in enumerate(zip(missing_pct.values, missing_pct.index)):
            plt.text(val + 0.3, i, f"{val:.1f}%", va='center', fontsize=9)

        plt.xlabel("Percentage of Missing Values (%)")
        plt.title(title, fontweight='bold')
        plt.xlim(0, missing_pct.max() * 1.15)
        plt.tight_layout()
        plt.show()

    # =========================================================================
    # 3. NUMERICAL FEATURE VISUALIZATIONS
    # =========================================================================

    def hist_frequencies(self, data, numeric_cols, bins=20, title="Distribution of Numerical Features"):
        """Plot histograms for all numerical features."""
        ncol_plots = 3
        nrow_plots = (len(numeric_cols) + ncol_plots - 1) // ncol_plots
        fig, axs = plt.subplots(nrow_plots, ncol_plots, figsize=(16, 4 * nrow_plots))
        axs = axs.flatten()

        colors = sbn.color_palette("husl", n_colors=len(numeric_cols))
        for i, col in enumerate(numeric_cols):
            sbn.histplot(data[col].dropna(), color=colors[i % len(colors)],
                         bins=bins, ax=axs[i], kde=True, edgecolor='white', linewidth=0.5)
            axs[i].set_title(f"Distribution of {col}", fontweight='bold')
            axs[i].set_xlabel(col)
            axs[i].set_ylabel("Frequency")

        # Hide unused subplots
        for j in range(i + 1, len(axs)):
            axs[j].set_visible(False)

        plt.suptitle(title, fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.show()

    def plot_boxplots(self, data, numeric_cols, title="Boxplot Analysis of Numerical Features"):
        """Plot boxplots for all numerical features to detect outliers."""
        ncol_plots = 3
        nrow_plots = (len(numeric_cols) + ncol_plots - 1) // ncol_plots
        fig, axs = plt.subplots(nrow_plots, ncol_plots, figsize=(16, 4 * nrow_plots))
        axs = axs.flatten()

        for i, col in enumerate(numeric_cols):
            sbn.boxplot(data=data, y=col, ax=axs[i], color=sbn.color_palette("Set2")[i % 8])
            axs[i].set_title(f"Boxplot of {col}", fontweight='bold')
            axs[i].set_ylabel(col)

        for j in range(i + 1, len(axs)):
            axs[j].set_visible(False)

        plt.suptitle(title, fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.show()

    def plot_violin(self, data, numeric_col, target_col, title=None):
        """Plot violin plot of a numerical feature grouped by target variable."""
        if title is None:
            title = f"{numeric_col} by {target_col}"
        plt.figure(figsize=(10, 6))
        order = sorted(data[target_col].dropna().unique())
        sbn.violinplot(data=data, x=target_col, y=numeric_col, order=order,
                       palette="Set2", inner="quartile")
        plt.title(title, fontweight='bold')
        plt.xlabel(target_col)
        plt.ylabel(numeric_col)
        plt.tight_layout()
        plt.show()

    # =========================================================================
    # 4. CORRELATION ANALYSIS
    # =========================================================================

    def plot_correlation(self, data, cols, title="Correlation Heatmap", method='pearson'):
        """Plot a correlation heatmap for the specified columns."""
        corr = data[cols].corr(method=method)

        plt.figure(figsize=(max(10, len(cols) * 0.6), max(8, len(cols) * 0.55)))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sbn.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, square=True, linewidths=0.5,
                    annot_kws={'size': 8}, vmin=-1, vmax=1)
        plt.title(title, fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.show()



    # =========================================================================
    # 5. CATEGORICAL FEATURE ANALYSIS
    # =========================================================================

    def get_categorical_instances(self, data, categ_cols):
        """Print value counts for each categorical column."""
        for col in categ_cols:
            print(f"\n{'='*50}")
            print(f"  {col}")
            print(f"{'='*50}")
            vc = data[col].value_counts(dropna=False)
            for val, count in vc.items():
                pct = count / len(data) * 100
                print(f"  {str(val):40s} : {count:>6,}  ({pct:.1f}%)")

    def plot_piechart(self, dataset, col, title=None):
        """Plot a pie chart for a single categorical column."""
        if title is None:
            title = f"Distribution of {col}"
        results = dataset[col].value_counts()
        total_samples = results.sum()

        plt.figure(figsize=(8, 8))
        colors = sbn.color_palette("Set2", n_colors=len(results))
        wedges, texts, autotexts = plt.pie(
            results.values, labels=results.index, autopct='%1.1f%%',
            colors=colors, pctdistance=0.8, startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        plt.title(title, fontweight='bold', fontsize=13)
        plt.tight_layout()
        plt.show()

    def iter_piechart(self, dataset, categ_cols, title="Categorical Feature Distributions"):
        """Plot multiple pie charts for categorical columns."""
        ncol_plots = 2
        nrow_plots = (len(categ_cols) + ncol_plots - 1) // ncol_plots
        fig, axs = plt.subplots(nrow_plots, ncol_plots, figsize=(14, 4 * nrow_plots))
        axs = axs.flatten()

        for i, col in enumerate(categ_cols):
            results = dataset[col].value_counts()
            colors = sbn.color_palette("Set2", n_colors=len(results))
            axs[i].pie(results.values, labels=results.index, autopct='%1.1f%%',
                       colors=colors, pctdistance=0.8, startangle=90,
                       wedgeprops={'edgecolor': 'white', 'linewidth': 1})
            axs[i].set_title(f"Distribution of {col}", fontweight='bold')

        for j in range(i + 1, len(axs)):
            axs[j].set_visible(False)

        plt.suptitle(title, fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.show()

    def plot_countplot(self, data, cat_col, target_col=None, title=None, rotation=45):
        """Plot count plot for a categorical variable, optionally grouped by target."""
        plt.figure(figsize=(10, 6))
        if target_col:
            sbn.countplot(data=data, x=cat_col, hue=target_col, palette="Set2")
        else:
            sbn.countplot(data=data, x=cat_col, palette="Set2")
        if title is None:
            title = f"Count Plot of {cat_col}" + (f" by {target_col}" if target_col else "")
        plt.title(title, fontweight='bold')
        plt.xlabel(cat_col)
        plt.ylabel("Count")
        plt.xticks(rotation=rotation, ha='right')
        plt.legend(title=target_col) if target_col else None
        plt.tight_layout()
        plt.show()

    # =========================================================================
    # 6. TARGET VARIABLE ANALYSIS
    # =========================================================================

    def plot_target_distribution(self, data, target, title="Target Variable Distribution"):
        """Plot distribution of the target variable (count + percentage)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Count plot
        order = sorted(data[target].dropna().unique())
        counts = data[target].value_counts().reindex(order)
        colors = sbn.color_palette("Set2", n_colors=len(order))

        sbn.barplot(x=counts.index, y=counts.values, palette=colors, ax=ax1, edgecolor='white')
        for i, (val, idx) in enumerate(zip(counts.values, counts.index)):
            ax1.text(i, val + len(data) * 0.005, f"{val:,}", ha='center', fontweight='bold', fontsize=10)
        ax1.set_title(f"Distribution of {target} (Counts)", fontweight='bold')
        ax1.set_ylabel("Count")
        ax1.set_xlabel(target)

        # Pie chart
        wedges, texts, autotexts = ax2.pie(
            counts.values, labels=counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        for at in autotexts:
            at.set_fontweight('bold')
        ax2.set_title(f"Distribution of {target} (Proportion)", fontweight='bold')

        plt.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    # =========================================================================
    # 7. SCATTER PLOTS & BIVARIATE ANALYSIS
    # =========================================================================

    def plot_scatter(self, data, x_col, y_col, target_col=None, title=None,
                     sample_size=None, alpha=0.3):
        """Scatter plot between two numerical features, colored by target."""
        if sample_size and len(data) > sample_size:
            plot_data = data.sample(sample_size, random_state=42)
        else:
            plot_data = data

        plt.figure(figsize=(10, 7))
        if target_col:
            for label in sorted(plot_data[target_col].dropna().unique()):
                mask = plot_data[target_col] == label
                plt.scatter(plot_data.loc[mask, x_col], plot_data.loc[mask, y_col],
                            label=label, alpha=alpha, s=15)
            plt.legend(title=target_col, markerscale=2)
        else:
            plt.scatter(plot_data[x_col], plot_data[y_col], alpha=alpha, s=15, c='steelblue')

        if title is None:
            title = f"{x_col} vs {y_col}"
        plt.title(title, fontweight='bold')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.tight_layout()
        plt.show()

    # =========================================================================
    # 8. DATA CLEANING UTILITIES
    # =========================================================================

    @staticmethod
    def clean_type_of_loan(frame):
        """
        Parse 'Type_of_Loan' column into binary indicator columns.
        Creates individual columns: auto_loan, credit_builder_loan, etc.
        """
        loan_types = [
            'Auto Loan', 'Credit-Builder Loan', 'Debt Consolidation Loan',
            'Home Equity Loan', 'Mortgage Loan', 'Not Specified',
            'Payday Loan', 'Personal Loan', 'Student Loan'
        ]
        # Generate safe column names
        safe_names = {
            'Auto Loan': 'auto_loan',
            'Credit-Builder Loan': 'credit_builder_loan',
            'Debt Consolidation Loan': 'debt_consolidation_loan',
            'Home Equity Loan': 'home_equity_loan',
            'Mortgage Loan': 'mortgage_loan',
            'Not Specified': 'not_specified_loan',
            'Payday Loan': 'payday_loan',
            'Personal Loan': 'personal_loan',
            'Student Loan': 'student_loan'
        }
        for loan, col_name in safe_names.items():
            frame[col_name] = frame['Type_of_Loan'].str.lower().str.contains(loan.lower()).fillna(False).astype(bool)
        frame.drop(columns=['Type_of_Loan'], inplace=True, errors='ignore')
        return frame

    @staticmethod
    def clean_credit_age(age_series):
        """Convert 'Credit_History_Age' from string 'X Years and Y Months' to float (years)."""
        def parse_age(age):
            if pd.isna(age) or age == 'nan' or not isinstance(age, str):
                return np.nan
            if "Years" not in age:
                try:
                    return float(age)
                except (ValueError, TypeError):
                    return np.nan
            try:
                years, months = age.split(" Years and ")
                months = months.replace(" Months", "")
                return int(years) + int(months) / 12
            except (ValueError, IndexError):
                return np.nan

        return age_series.apply(parse_age)

    @staticmethod
    def clean_payment_behaviour(frame):
        """Fix the anomalous value '!@9#%8' in Payment_Behaviour column."""
        frame['Payment_Behaviour'] = frame['Payment_Behaviour'].replace('!@9#%8', np.nan)
        return frame

    @staticmethod
    def clean_outliers(frame):
        """
        Handle outliers in known columns:
        - Age > 65 → cap at 65
        - Num_Bank_Accounts > 1000 → cap at 1000
        - Monthly_Balance > 1e6 → set to NaN
        """
        frame = frame.copy()
        frame.loc[frame["Age"] > 65, "Age"] = 65
        frame.loc[frame["Age"] < 0, "Age"] = np.nan
        frame.loc[frame["Num_Bank_Accounts"] > 1000, "Num_Bank_Accounts"] = 1000
        frame.loc[frame["Monthly_Balance"] > 1e6, "Monthly_Balance"] = np.nan

        # Refined outlier detection using robust bounds
        for col in ["Monthly_Balance", "Amount_invested_monthly", "Outstanding_Debt"]:
            if col in frame.columns and frame[col].dtype in ['float64', 'int64']:
                Q1 = frame[col].quantile(0.25)
                Q3 = frame[col].quantile(0.75)
                IQR = Q3 - Q1
                upper_bound = Q3 + 3 * IQR  # Using 3 for extreme outliers
                lower_bound = Q1 - 3 * IQR
                
                # We replace extreme outliers with NaN, which will be imputed later
                frame.loc[frame[col] > upper_bound, col] = np.nan
                frame.loc[frame[col] < lower_bound, col] = np.nan

        return frame

    @staticmethod
    def clean_numeric_strings(frame, numeric_object_columns):
        """Convert columns that are numeric but stored as object (remove non-numeric chars)."""
        for col in numeric_object_columns:
            if col in frame.columns:
                frame[col] = frame[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
                frame[col] = pd.to_numeric(frame[col], errors='coerce')
        return frame

    @staticmethod
    def clean_credit_mix(frame):
        """Fix '_' values in Credit_Mix column → replace with NaN."""
        frame['Credit_Mix'] = frame['Credit_Mix'].replace('_', np.nan)
        return frame

    @staticmethod
    def clean_occupation(frame):
        """Fix '_______' values in Occupation column → replace with NaN."""
        frame['Occupation'] = frame['Occupation'].replace('_______', np.nan)
        return frame
