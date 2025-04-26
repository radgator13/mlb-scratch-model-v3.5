import subprocess

# List of scripts to run
scripts = [
    "get_scores_full.py",
    "build_stats.py",
    "merge_stats.py",
    "build_nrfi_results.py"
]

print("\n🚀 Starting Full Data Refresh...")

for script in scripts:
    print(f"\n▶️ Running: {script}")
    subprocess.run(["python", script], check=True)

print("\n✅ Full Refresh Complete! You can now launch your app with:")
print("\n    streamlit run app.py")
