# TapDance

See a demo here!
<br>
[![Youtube Video: https://www.youtube.com/watch?v=mOv5McYPgPM](https://img.youtube.com/vi/3BvJOD-v0QU/0.jpg)](https://youtu.be/3BvJOD-v0QU)

## Inspiration
I was inspired by the idea of making exercise fun and interactive, especially since just plainly working out is not fun for some people. By combining dance, body movement, and computer vision, TapDance is a game that encourages physical activity while being fun (and very entertaining to watch).

## How to play
TapDance is an AI-powered dance game that uses pose detection to track a player’s hands and feet through a webcam. Match both hands to the correspondingly coloured target (red to red, green to green, etc) at the same time to gain points and climb the leaderboard. There is a hands-only mode and hands-and-feet mode. 

## How it's made
- Used Python with OpenCV for real-time video processing
- Integrated MediaPipe Pose for detecting body landmarks
- Designed the UI for the screens (menu, name input, leaderboard, game over) using Figma
- Made the screens interactive using OpenCV

## Challenges
- Combining the "hands only" and "hands and feet" codes together as they were originally put as 2 separate files as I was testing
- Having the dots show and move correctly after being hit
- Getting the leaderboard to work correctly

## Accomplishments
- Using OpenCV and MediaPipe for the first time
- A fully functional, camera-based dance game with both hands-only and hands+feet modes
- A working leaderboard
  
## What I learned
- How to integrate pose estimation into real-time video applications.
- Advanced use of OpenCV for custom UI design (buttons, overlays, text inputs).
- Structuring a Python project for interactivity and smooth UX.
- Techniques for improving detection robustness in real-time environments.
  
## What's next for TapDance
- Adding multiplayer support
- Having difficulty levels and dynamic target movement
- Different gamemodes - reaction time test, multiplayer, moving targets, rush mode where the targets move across the screen instead
- Making it into a mobile app using device cameras
- Including music that matches the beats
- Using machine learning to generate dance patterns based on player behavior
