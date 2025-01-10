import os
import subprocess
import sys

def clear_console():
    """Clear the console for better user experience."""
    os.system('cls' if os.name == 'nt' else 'clear')

def main_menu():
    """Display the main menu and handle user choices."""
    while True:
        clear_console()
        print("====== Facial Recognition System ======")
        print("1. Capture a new person's face")
        print("2. Train the model with captured faces")
        print("3. Perform live facial recognition")
        print("4. Exit")
        print("=======================================")
        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            capture_faces()
        elif choice == "2":
            train_model()
        elif choice == "3":
            live_detection()
        elif choice == "4":
            print("Exiting... Goodbye!")
            break
        else:
            print("Invalid choice! Please try again.")
            input("Press Enter to continue...")

def capture_faces():
    """Run the script to capture faces."""
    clear_console()
    print("====== Capture Faces ======")

    # Use the .venv Python interpreter to run the script
    subprocess.run([sys.executable, "scripts/capture_faces.py"], check=True)
    print(f"Face data captured successfully.")
    input("Press Enter to return to the main menu...")

def train_model():
    """Run the script to train the model."""
    clear_console()
    print("====== Train Model ======")
    print("Training the model with the captured faces...")
    
    # Use the .venv Python interpreter to run the script
    subprocess.run([sys.executable, "scripts/train_model.py"], check=True)
    print("Model training completed successfully.")
    input("Press Enter to return to the main menu...")

def live_detection():
    """Run the script to perform live facial recognition."""
    clear_console()
    print("====== Live Facial Recognition ======")
    print("Starting live detection. Press 'q' to quit.")
    
    # Use the .venv Python interpreter to run the script
    subprocess.run([sys.executable, "scripts/recognize_faces.py"], check=True)
    print("Exiting live detection.")
    input("Press Enter to return to the main menu...")

if __name__ == "__main__":
    main_menu()
