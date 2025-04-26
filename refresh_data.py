import subprocess
import datetime

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
