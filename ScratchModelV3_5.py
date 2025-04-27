import subprocess
import datetime
import os

# List of scripts to run DAILY (no retraining)
scripts = [
    "get_scores_full.py",
    "build_stats.py",
    "merge_stats.py",
    "build_nrfi_results.py"
]

print("\n🚀 Starting Full Data Refresh...")

# Run each script
for script in scripts:
    print(f"\n▶️ Running: {script}")
    subprocess.run(["python", script], check=True)

print("\n✅ Data Refresh Complete!")

# ------------------------------
# Git Add / Commit / Push
# ------------------------------

print("\n🚀 Starting Git Push to GitHub...")

timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
commit_message = f"Auto-Refresh: Updated data only ({timestamp})"

try:
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", commit_message], check=True)
    subprocess.run(["git", "push"], check=True)
    print("\n✅ Git Push Complete! Everything is updated on GitHub.")
except subprocess.CalledProcessError as e:
    print("\n❌ Git Push Failed! Error details below:")
    print(e)

print("\n🎯 You can now launch your app with:")
print("\n    streamlit run app.py")

# ------------------------------
# 📋 Final File Check
# ------------------------------

import os

print("\n🚀 Checking all critical files...")

# List of critical files
critical_files = [
    "data/mlb_boxscores_full.csv",
    "data/mlb_nrfi_predictions.csv",
    "data/mlb_nrfi_results_full.csv",
]

# Flag to track success
pipeline_success = True

# Check each file
for file in critical_files:
    if not os.path.exists(file):
        print(f"❌ Missing file: {file}")
        pipeline_success = False
    else:
        df_check = pd.read_csv(file)
        if df_check.empty:
            print(f"❌ File exists but is EMPTY: {file}")
            pipeline_success = False
        else:
            print(f"✅ File OK: {file} ({len(df_check)} rows)")

if pipeline_success:
    print("\n✅ Pipeline Check PASSED! All files ready.")
else:
    print("\n❌ Pipeline Check FAILED! Some files are missing or empty.")

print("\n🎯 Ready to launch dashboard if all checks are green!")
