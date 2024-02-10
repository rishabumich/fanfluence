import streamlit as st
import time

# Define your list of events
events = ["Chick-fil-A Peach Bowl", "SEC Championship", "Super Bowl LIII", "MLS All-Star Game", "Celebration Bowl"]

best_worst_tweets = {
    "Atmosphere": {"best": "Amazing atmosphere at the game!", "worst": "The atmosphere could have been better."},
    "Food": {"best": "The food was delicious!", "worst": "Long lines for food, not a great experience."},
    "Security": {"best": "Felt very secure throughout the event.", "worst": "Security checks were too slow."},
    "Cleanliness": {"best": "The stadium was very clean!", "worst": "Trash bins were overflowing."}
}

def reset_animation_state():
    for event in events:
        st.session_state.animation_played[event] = False
    for category in best_worst_tweets.keys():
        st.session_state.animation_played[category] = False

def main():
    # Initialize or update session state for page navigation and event selection
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'selected_event' not in st.session_state:
        st.session_state.selected_event = ''
    if 'animation_played' not in st.session_state:
        st.session_state.animation_played = {}

    # Page navigation logic
    if st.session_state.page == 'home':
        display_home_page()
    elif st.session_state.page == 'event_details':
        display_event_details_page()

def display_home_page():
    """Displays the home page with event selection."""
    st.markdown("<h2 style='text-align: center; color: White;'>Welcome To The Benz Stadium </h1>", unsafe_allow_html=True)

    st.image('/Users/PeddiChittepu/Downloads/AAE/Benz.jpeg', use_column_width=True)


    event = st.selectbox("Choose an event:", events)
    st.session_state.selected_event = event
    if st.button("Continue"):
        st.session_state.page = 'event_details'
        reset_animation_state()

def display_event_details_page():
    """Displays the event details page with animations and score markers."""
    st.header(st.session_state.selected_event)
    scores = {"Atmosphere": 1, "Food": 1, "Security": 1, "Cleanliness": 1}

    event = st.session_state.selected_event
    for score_name, score_value in scores.items():
        if not st.session_state.animation_played.get(score_name, False):
            st.subheader(score_name)
            progress_bar = st.progress(0)
            for percent_complete in range(score_value * 20 + 1):
                time.sleep(0.02)
                progress_bar.progress(percent_complete)
            st.session_state.animation_played[score_name] = True  # Mark this category's animation as done
            
            # Display the best and worst tweets for the category
            st.markdown("**Best Tweet:** " + best_worst_tweets[score_name]["best"])
            st.markdown("**Worst Tweet:** " + best_worst_tweets[score_name]["worst"])
            
    if st.button("Back to Home"):
        # Resetting the page state to home and re-running the app
        st.session_state.page = 'home'
        reset_animation_state()
        st.experimental_rerun()

if __name__ == "__main__":
    main()
