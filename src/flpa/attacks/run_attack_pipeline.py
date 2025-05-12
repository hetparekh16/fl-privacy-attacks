import subprocess

print("\n🛠️  [1/3] Building membership dataset...")
subprocess.run(["python", "src/flpa/attacks/build_membership_dataset.py"], check=True)

print("\n📈 [2/3] Extracting posteriors using the global model...")
subprocess.run(["python", "src/flpa/attacks/extract_posteriors.py"], check=True)

print("\n🧠 [3/3] Training attack models (logistic, random forest, MLP)...")
subprocess.run(["python", "src/flpa/attacks/train_attack_model.py"], check=True)

print("\n✅ All attack steps completed successfully!")
