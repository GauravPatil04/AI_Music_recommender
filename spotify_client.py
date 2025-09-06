import os
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth

class SpotifyClient:
    """
    Spotify client that loads credentials from a .env file, handles authentication,
    and fetches song recommendations based on emotion.
    """
    def __init__(self):
        """Initializes the client and authenticates with Spotify."""
        load_dotenv()

        self.client_id = os.getenv("SPOTIPY_CLIENT_ID")
        self.client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
        self.redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI")

        if not all([self.client_id, self.client_secret, self.redirect_uri]):
            raise ValueError("Missing credentials in .env file.")

        try:
            auth_manager = SpotifyOAuth(
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=self.redirect_uri,
                scope="user-read-private",
                cache_path=".spotify_cache"
            )
            self.sp: Spotify = Spotify(auth_manager=auth_manager)
            user_info = self.sp.current_user()
            self.market: str = user_info.get('country', 'US')
            logging.info(f"✅ Authenticated! Market: {self.market}")

            # Correct endpoint for genre seeds
            genre_data = self.sp.recommendation_genre_seeds()
            self.valid_genres = set(genre_data.get('genres', []))

        except Exception as e:
            raise ConnectionError(f"Spotify authentication failed: {str(e)}")

    def get_recommendations(self, emotion: str, limit: int = 5) -> Optional[List[Dict]]:
        """
        Gets track recommendations from Spotify based on emotion.
        """
        mood_config = {
            'happy': {
                'seed_genres': ['pop', 'dance', 'electronic'],
                'target_valence': 0.8,
                'target_energy': 0.8
            },
            'sad': {
                'seed_genres': ['acoustic', 'piano', 'rainy-day'],
                'target_valence': 0.2,
                'target_energy': 0.3
            },
            'angry': {
                'seed_genres': ['metal', 'rock', 'hard-rock'],
                'target_valence': 0.3,
                'target_energy': 0.9
            },
            'neutral': {
                'seed_genres': ['chill', 'lo-fi', 'ambient'],
                'target_valence': 0.5,
                'target_energy': 0.5
            }
        }

        emotion = emotion.lower()
        config = mood_config.get(emotion)
        if not config:
            logging.warning(f"Unsupported emotion '{emotion}'. Try one of: {list(mood_config.keys())}")
            return None

        # Validate genres
        valid_seeds = [g for g in config['seed_genres'] if g in self.valid_genres]
        if not valid_seeds:
            logging.error(f"All seed genres for '{emotion}' are invalid.")
            return None

        try:
            results = self.sp.recommendations(
                limit=limit,
                market=self.market,
                seed_genres=valid_seeds,
                target_valence=config['target_valence'],
                target_energy=config['target_energy']
            )
            return results.get('tracks', [])
        except Exception as e:
            logging.error(f"Error fetching recommendations: {str(e)}")
            return None

# 🔍 Test block
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        logging.info("🚀 Launching Spotify Mood Recommender...")
        client = SpotifyClient()

        # Replace this with the output from your emotion model
        emotion_detected = 'happy'
        tracks = client.get_recommendations(emotion_detected, limit=5)
        # client = SpotifyClient()
        genres = client.sp.recommendation_genre_seeds()
        print(genres['genres'])


        if tracks:
            print(f"\n🎵 Top tracks for '{emotion_detected}' mood:")
            for i, track in enumerate(tracks, 1):
                name = track.get("name")
                artists = ", ".join(artist["name"] for artist in track.get("artists", []))
                print(f"{i}. {name} by {artists}")
        else:
            print("😞 No recommendations found.")
    except Exception as e:
        logging.error(f"💥 Fatal error: {str(e)}")
