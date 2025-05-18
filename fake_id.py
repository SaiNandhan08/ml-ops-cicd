# insta_fake_detector.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import instaloader
import re
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, VotingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
   

# Helper function to check for suspicious links
SUSPICIOUS_DOMAIN_PATTERNS = [
    "thepiratebay", "1337x", "rarbg", "torrentz2", # Example torrent sites
    "fmovies", "putlocker", "yesmovies" # Example streaming sites often associated with piracy
]

def is_link_suspicious(url: str) -> bool:
    if not url.startswith(("http://", "https://")):
        url = "http://" + url # Ensure re.search can find domain properly
    for pattern in SUSPICIOUS_DOMAIN_PATTERNS:
        # Search for pattern in the domain part of the URL
        # This is a simplified check and might need more robust URL parsing for edge cases
        if re.search(r"https?://([^/?#]+).*", url):
            domain = re.search(r"https?://([^/?#]+).*", url).group(1)
            if pattern in domain:
                print(f"Suspicious pattern '{pattern}' found in URL: {url}")
                return True
    return False

# Set Seaborn theme
sns.set_theme()

# --------------------
# Load data
# --------------------
def load_data(train_url: str, test_url: str):
    train = pd.read_csv(train_url)
    test = pd.read_csv(test_url)
    return train, test

# --------------------
# Plot correlation heatmap
# --------------------
def plot_correlation(df):
    plt.figure(figsize=(12, 12))
    sns.heatmap(df.corr(), cmap='coolwarm', linewidths=0.9)
    plt.title("Feature Correlation Heatmap")
    plt.show()

# --------------------
# Plot class distribution
# --------------------
def plot_class_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x="fake", data=df, palette="Set2")
    plt.title("Fake vs Real Instagram Profiles")
    plt.xlabel("Is Fake?")
    plt.ylabel("Count")
    plt.show()

# --------------------
# Train and evaluate models
# --------------------

def train_models(X_train, y_train, X_test, y_test):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=30, max_depth=20, max_samples=350, random_state=28),
        "Linear SVC": LinearSVC(random_state=28),
        "KNN": KNeighborsClassifier(n_neighbors=10, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "MLP": MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 12), random_state=28),
        "Decision Tree": DecisionTreeClassifier(criterion="gini", max_depth=2, random_state=42),
        "AdaBoost": AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=5, learning_rate=1)
    }
 
    for name, model in models.items():
          model.fit(X_train, y_train)
          pred = model.predict(X_test)
          print(f"\n📌 {name}")
          print("Accuracy:", accuracy_score(y_test, pred))
          print(classification_report(y_test, pred))

def scrape_profile_features(instagram_url: str):
    loader = instaloader.Instaloader()

    match = re.search(r"instagram\.com/([^/?]+)", instagram_url)
    if not match:
        print("⚠️ Invalid Instagram URL")
        return None
    username = match.group(1)
    # print(f"Scraping profile: {username}") # Optional: for debugging

    try:
        profile = instaloader.Profile.from_username(loader.context, username)
        
        # Check for suspicious links in bio first
        if profile.biography:
            urls_in_bio = re.findall(r'(https?://[\w\d./?=#&%-]+|www\.[\w\d./?=#&%-]+|\b[\w\d.-]+\.(?:com|org|net|io|co|info|biz|me|xyz)(?:/[\w\d./?=#&%-]*)?)\b', profile.biography, re.IGNORECASE)
            for bio_url in urls_in_bio:
                if is_link_suspicious(bio_url):
                    print(f"Profile {username} flagged due to suspicious link in bio: {bio_url}")
                    return "SUSPICIOUS_LINK_FOUND"

        # Feature 1: Profile Picture
        has_profile_picture = 0 if profile.profile_pic_url is None or profile.profile_pic_url == "" else 1
        
        # Feature 2: Nums/Length Username
        username_len = len(username)
        nums_in_username = sum(c.isdigit() for c in username)
        nums_by_length_username = nums_in_username / username_len if username_len > 0 else 0
        
        # Feature 3: Fullname Words
        full_name_words = len(profile.full_name.split()) if profile.full_name else 0
        
        # Feature 4: Nums/Length Fullname
        fullname_len = len(profile.full_name) if profile.full_name else 0
        nums_in_fullname = sum(c.isdigit() for c in profile.full_name) if profile.full_name else 0
        nums_by_length_fullname = nums_in_fullname / fullname_len if fullname_len > 0 else 0
        
        # Feature 5: Name == Username (simple check)
        # A more sophisticated check might be needed depending on dataset's definition
        name_eq_username = 1 if profile.full_name and username.lower() == profile.full_name.lower().replace(" ", "") else 0
        
        # Feature 6: Description Length
        description_length = len(profile.biography) if profile.biography else 0
        
        # Feature 7: External URL present in profile's dedicated field (not just bio text)
        external_url_present = 1 if profile.external_url and len(profile.external_url) > 0 else 0
        
        # Feature 8: Is Private
        is_private = 1 if profile.is_private else 0
        
        # Feature 9: Posts Count
        posts_count = profile.mediacount
        
        # Feature 10: Followers Count
        followers_count = profile.followers
        
        # Feature 11: Follows Count
        followees_count = profile.followees

        feature_vector = [
            float(has_profile_picture),
            float(nums_by_length_username),
            float(full_name_words),
            float(nums_by_length_fullname),
            float(name_eq_username),
            float(description_length),
            float(external_url_present),
            float(is_private),
            float(posts_count),
            float(followers_count),
            float(followees_count)
        ]
        # print(f"Generated features for {username}: {feature_vector}") # Optional: for debugging
        return feature_vector

    except instaloader.exceptions.ProfileNotExistsException:
        print(f"❌ Profile {username} does not exist.")
        return None
    except Exception as e:
        print(f"❌ Error scraping profile {username}: {e}")
        return None

# --------------------
# Voting Ensemble
# --------------------
def run_ensemble(X_train, y_train, X_test, y_test):
    models = [
        ('mlp', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 12), random_state=28)),
        ('dt', DecisionTreeClassifier()),
        ('svm', LinearSVC()),
        ('rf', RandomForestClassifier()),
        ('nb', GaussianNB()),
        ('kn', KNeighborsClassifier(n_neighbors=10, weights="distance"))
    ]

    voting_clf = VotingClassifier(estimators=models, voting='hard')
    voting_clf.fit(X_train, y_train)
    pred = voting_clf.predict(X_test)
    
    print("\n🗳️ Voting Classifier")
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))
    
    # Calculate and print False Negatives
    # Assuming class 1 is 'fake' and class 0 is 'real'
    # TN, FP, FN, TP
    cm = confusion_matrix(y_test, pred)
    true_negatives = cm[0][0] if cm.shape == (2,2) else "N/A (check classes)"
    true_positives = cm[1][1] if cm.shape == (2,2) else "N/A (check classes)"
    print(f"True Negatives (Real profiles predicted as Real): {true_negatives}")
    print(f"True Positives (Fake profiles correctly predicted as Fake): {true_positives}")
    
    return voting_clf

# --------------------
# Run the entire flow
# --------------------
def main():
    # Replace with local paths if running offline
    train_url = "https://www.dropbox.com/s/uanezjf9y1xb2te/train.csv?dl=1"
    test_url = "https://www.dropbox.com/s/ap90v0bu9td4k4y/test.csv?dl=1"

    df_train, df_test = load_data(train_url, test_url)

    print("Nulls in Train:\n", df_train.isnull().sum())
    print("Nulls in Test:\n", df_test.isnull().sum())
    print("Correlation with 'fake':\n", df_train.corr()[['fake']])

    # plot_correlation(df_train)  # Commented out to prevent display
    # plot_class_distribution(df_train)  # Commented out to prevent display

    X_train = df_train.drop("fake", axis=1)
    y_train = df_train["fake"]
    X_test = df_test.drop("fake", axis=1)
    y_test = df_test["fake"]

    train_models(X_train, y_train, X_test, y_test)

    # Voting Ensemble
#    voting_model = 
    run_ensemble(X_train, y_train, X_test, y_test)

    # Predict one sample
'''   print("\n🔗 Paste the Instagram profile URL to check:")
    ig_url = input()

    scraped_data = scrape_profile_features(ig_url)
    
    if scraped_data == "SUSPICIOUS_LINK_FOUND":
        result = "Fake ID (Suspicious link in bio)"
        print(f"\n🎯 Prediction for {ig_url}: {result}")
    elif scraped_data is not None: # Check if it's a feature list
        features = scraped_data
        prediction = voting_model.predict([features])
        # Your existing logic for fake vs real based on prediction[0]
        # Assuming 0 is Real and 1 is Fake
        result = "Real ID " if prediction[0] == 0 else "Fake ID"
        print(f"\n🎯 Prediction for {ig_url}: {result}")
    else:
        # scrape_profile_features returned None (e.g., error or invalid URL)
        print(f"Could not retrieve features for {ig_url}.") '''

if __name__ == "__main__":
    main()

