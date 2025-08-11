
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

plt.ioff()

class OptimizedBiasAnalyzer:
    def __init__(self, log_file_path='experiment_log.json', output_dir='results'):
        """Initialize optimized SHAP analyzer"""
        self.output_dir = Path(output_dir)
        self.setup_output_directories()
        self.load_data(log_file_path)
        self.results = {}

        self.random_state = 42
        self.min_samples = 30  
        self.max_features = 2000  
        self.min_df = 2  
        self.max_df = 0.95  

    def setup_output_directories(self):
        """Create output directories"""
        directories = [
            self.output_dir,
            self.output_dir / 'subcategory_plots',
            self.output_dir / 'detailed_results'
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        print(f"Output directories created in: {self.output_dir}")

    def load_data(self, log_file_path):
        """Load experimental data"""
        print("Loading experimental data...")
        with open(log_file_path, 'r') as f:
            self.raw_data = json.load(f)

        self.df = pd.DataFrame(self.raw_data)
        # Binary encoding: 0 = Female (A), 1 = Male (B)
        self.df['gender_label'] = (self.df['chosen_option'] == 'B').astype(int)

        print(f"Loaded {len(self.df)} responses")
        print(f"Subcategories: {self.df['subcategory'].nunique()}")

    def create_optimal_vectorizer(self, texts):
        """Create optimized vectorizer based on data characteristics"""
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=(1, 1),  
            lowercase=True,
            strip_accents='ascii'
        )
        return vectorizer

    def analyze_subcategory(self, subcategory):
        """Enhanced subcategory analysis with proper SHAP implementation"""
        sub_data = self.df[self.df['subcategory'] == subcategory].copy()

        if len(sub_data) < self.min_samples:
            print(f"Insufficient data for {subcategory}: {len(sub_data)} samples")
            return None

        print(f"\nAnalyzing: {subcategory} ({len(sub_data)} samples)")

        class_counts = sub_data['gender_label'].value_counts()
        if len(class_counts) < 2 or min(class_counts) < 5:
            print(f"Skipping {subcategory}: Insufficient class balance")
            print(class_counts)
            return None

        vectorizer = self.create_optimal_vectorizer(sub_data['prompt'])
        X = vectorizer.fit_transform(sub_data['prompt'])
        y = sub_data['gender_label'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        classifier = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            C=1.0,  
            class_weight='balanced'
        )

        classifier.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, classifier.predict(X_train))
        test_acc = accuracy_score(y_test, classifier.predict(X_test))

        print(f"Model accuracy - Train: {train_acc:.3f}, Test: {test_acc:.3f}")

        explainer = shap.LinearExplainer(classifier, X_train)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            # For binary classification, take the positive class (Male = 1)
            shap_values_final = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        else:
            shap_values_final = shap_values

        print(f"SHAP values shape: {shap_values_final.shape}")

        feature_names = vectorizer.get_feature_names_out()

        mean_shap = np.mean(shap_values_final, axis=0)
        std_shap = np.std(shap_values_final, axis=0)

        mean_shap = np.nan_to_num(mean_shap, nan=0.0, posinf=0.0, neginf=0.0)
        std_shap = np.nan_to_num(std_shap, nan=0.0, posinf=0.0, neginf=0.0)

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_shap': mean_shap,
            'std_shap': std_shap,
            'abs_shap': np.abs(mean_shap),
            'sample_count': len(sub_data)
        }).sort_values('abs_shap', ascending=False)

        female_count = (sub_data['chosen_option'] == 'A').sum()
        male_count = (sub_data['chosen_option'] == 'B').sum()
        bias_score = abs(male_count - female_count) / len(sub_data)
        bias_direction = 'Male' if male_count > female_count else 'Female'

        result = {
            'subcategory': subcategory,
            'total_samples': len(sub_data),
            'female_count': female_count,
            'male_count': male_count,
            'bias_score': bias_score,
            'bias_direction': bias_direction,
            'model_accuracy': test_acc,
            'importance_df': importance_df,
            'class_balance': min(class_counts) / max(class_counts)
        }

        self.create_enhanced_plot(result)

        self.save_detailed_results(result)

        return result

    def create_enhanced_plot(self, result):
        """Create publication-quality SHAP visualization"""
        subcategory = result['subcategory']
        importance_df = result['importance_df']

        try:
            top_features = importance_df.head(15)

            features = top_features['feature'].tolist()
            values = top_features['mean_shap'].tolist()

            colors = ['#4472C4' if v > 0 else '#E91E63' for v in values]  # Professional blue and pink

            fig, ax = plt.subplots(figsize=(12, 8))

            y_pos = np.arange(len(features))
            bars = ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=12)
            ax.set_xlabel('SHAP Value (Token Influence)', fontsize=14, fontweight='bold')
            ax.set_title(f'{subcategory.replace("_", " ").title()}: Top Tokens by SHAP Value',
                        fontsize=16, fontweight='bold', pad=20)

            ax.axvline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)

            ax.invert_yaxis()

            ax.text(0.02, 0.98, 'Female Bias ←', transform=ax.transAxes,
                   fontsize=12, fontweight='bold', color='#E91E63',
                   verticalalignment='top')
            ax.text(0.98, 0.98, '→ Male Bias', transform=ax.transAxes,
                   fontsize=12, fontweight='bold', color='#4472C4',
                   verticalalignment='top', horizontalalignment='right')

            bias_direction = result['bias_direction']
            bias_score = result['bias_score']
            subtitle = f"Samples: {result['total_samples']} | Bias Score: {bias_score:.3f} ({bias_direction})"
            ax.text(0.5, 0.02, subtitle, transform=ax.transAxes,
                   fontsize=11, ha='center', style='italic')

            plt.tight_layout()
            plt.grid(axis='x', alpha=0.3, linestyle='--')

            filename = f"{subcategory.replace(' ', '_').lower()}_shap_analysis.png"
            filepath = self.output_dir / 'subcategory_plots' / filename
            plt.savefig(str(filepath), dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()

            print(f"Enhanced plot saved: {filename}")

        except Exception as e:
            print(f"Plot error for {subcategory}: {e}")
            plt.close()

    def save_detailed_results(self, result):
        """Save detailed token analysis"""
        subcategory = result['subcategory']
        importance_df = result['importance_df']

        detailed_results = importance_df.head(50).copy()
        detailed_results['subcategory'] = subcategory
        detailed_results['bias_direction'] = result['bias_direction']
        detailed_results['bias_score'] = result['bias_score']

        filename = f"{subcategory.replace(' ', '_').lower()}_detailed_tokens.csv"
        filepath = self.output_dir / 'detailed_results' / filename
        detailed_results.to_csv(filepath, index=False)

    def analyze_all_subcategories(self):
        """Analyze all subcategories with progress tracking"""
        print("\nStarting comprehensive SHAP analysis...")

        subcategories = self.df['subcategory'].unique()
        self.results = {}
        successful_analyses = 0

        for subcategory in tqdm(subcategories, desc="Processing subcategories"):
            result = self.analyze_subcategory(subcategory)
            if result:
                self.results[subcategory] = result
                successful_analyses += 1

        print(f"\nCompleted {successful_analyses}/{len(subcategories)} subcategories successfully")
        return self.results

    def create_summary_report(self):
        """Generate comprehensive summary report"""
        print("\nGenerating summary report...")

        # Comprehensive summary data
        summary_data = []
        for name, result in self.results.items():
            top_tokens = result['importance_df'].head(3)

            summary_data.append({
                'subcategory': name,
                'bias_score': result['bias_score'],
                'bias_direction': result['bias_direction'],
                'total_samples': result['total_samples'],
                'female_count': result['female_count'],
                'male_count': result['male_count'],
                'model_accuracy': result['model_accuracy'],
                'class_balance': result['class_balance'],
                'top_token_1': top_tokens.iloc[0]['feature'],
                'top_token_1_shap': top_tokens.iloc[0]['mean_shap'],
                'top_token_2': top_tokens.iloc[1]['feature'],
                'top_token_2_shap': top_tokens.iloc[1]['mean_shap'],
                'top_token_3': top_tokens.iloc[2]['feature'],
                'top_token_3_shap': top_tokens.iloc[2]['mean_shap']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('bias_score', ascending=False)

        # Save comprehensive summary
        summary_df.to_csv(self.output_dir / 'detailed_results' / 'comprehensive_summary.csv', index=False)

        # Print key insights
        print(f"\n{'='*60}")
        print("KEY FINDINGS FROM SHAP ANALYSIS")
        print(f"{'='*60}")

        bias_scores = [r['bias_score'] for r in self.results.values()]
        accuracies = [r['model_accuracy'] for r in self.results.values()]

        print(f"Overall Statistics:")
        print(f"  • Average bias score: {np.mean(bias_scores):.3f}")
        print(f"  • Average model accuracy: {np.mean(accuracies):.3f}")
        print(f"  • Subcategories analyzed: {len(self.results)}")

        print(f"\nMost Biased Subcategories:")
        for i, (_, row) in enumerate(summary_df.head(5).iterrows(), 1):
            print(f"  {i}. {row['subcategory'].replace('_', ' ').title()}")
            print(f"     Bias: {row['bias_score']:.3f} (→ {row['bias_direction']})")
            print(f"     Top token: '{row['top_token_1']}' (SHAP: {row['top_token_1_shap']:.3f})")

        print(f"\nAll detailed results saved to: {self.output_dir}")

def main():
    """Main execution function"""
    print("Starting Optimized SHAP Gender Bias Analysis")
    print("="*60)

    analyzer = OptimizedBiasAnalyzer('experiment_log.json')

    results = analyzer.analyze_all_subcategories()

    if results:

        analyzer.create_summary_report()

        print(f"\n🎉 Analysis completed successfully!")
        print(f"📊 Results available in: {analyzer.output_dir}")
    else:
        print("❌ No successful analyses completed")

    return analyzer

if __name__ == "__main__":
    analyzer = main()


