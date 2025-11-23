import os
import matplotlib.pyplot as plt
from src.data_loader import DarwinDataLoader
from src.model import create_alzheimers_model

# Configuration
DATA_PATH = "data/data.csv"
MODEL_SAVE_PATH = "models/alzheimer_model.keras"

def main():
    # 1. Prepare Data
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: File not found at {DATA_PATH}. Please add the DARWIN CSV file.")
        return

    loader = DarwinDataLoader(DATA_PATH)
    X_train, X_test, y_train, y_test = loader.get_split_data()
    
    print(f"Data Loaded. Training Shape: {X_train.shape}")

    # 2. Build Model
    model = create_alzheimers_model(input_dim=X_train.shape[1])
    model.summary()

    # 3. Train
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # 4. Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Final Test Accuracy: {acc:.4f}")
    print(f"❌ Final Test Loss: {loss:.4f}")

    # 5. Save Model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # Optional: Plot history
    plot_history(history)

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig("training_results.png")
    print("Training graph saved as training_results.png")

if __name__ == "__main__":
    main()
