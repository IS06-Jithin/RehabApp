import React, { useRef, useEffect, useState } from 'react';
import { Pose } from '@mediapipe/pose';
import { Camera } from '@mediapipe/camera_utils';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const [feedback, setFeedback] = useState('');
  const [exerciseStarted, setExerciseStarted] = useState(false);
  const [progressData, setProgressData] = useState({ labels: [], datasets: [] });

  // Initialize WebSocket connection
  useEffect(() => {
    wsRef.current = new WebSocket('ws://localhost:8000/ws');
    wsRef.current.onopen = () => console.log('WebSocket Connected');
    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setFeedback(data.message);
      if (data.deviations) {
        // Update progress data (example: track torso deviation)
        setProgressData({
          labels: [...progressData.labels, new Date().toLocaleTimeString()],
          datasets: [{
            label: 'Torso Deviation',
            data: [...(progressData.datasets[0]?.data || []), data.deviations.torso || 0],
            backgroundColor: 'rgba(75, 192, 192, 0.5)',
          }],
        });
      }
      // Auditory feedback using Web Speech API
      if (data.message) {
        const utterance = new SpeechSynthesisUtterance(data.message);
        speechSynthesis.speak(utterance);
      }
    };
    return () => wsRef.current.close();
  }, [progressData]);

  // Setup MediaPipe Pose
  useEffect(() => {
    if (!exerciseStarted) return;

    const pose = new Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });
    pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    pose.onResults((results) => {
      // Draw keypoints on canvas for visualization
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (results.poseLandmarks) {
        results.poseLandmarks.forEach((landmark) => {
          const x = landmark.x * canvas.width;
          const y = landmark.y * canvas.height;
          ctx.beginPath();
          ctx.arc(x, y, 5, 0, 2 * Math.PI);
          ctx.fillStyle = 'red';
          ctx.fill();
        });
        // Send keypoints to backend
        if (wsRef.current.readyState === WebSocket.OPEN) {
          const keypoints = results.poseLandmarks.map((landmark) => [
            landmark.x,
            landmark.y,
            landmark.z,
          ]);
          wsRef.current.send(JSON.stringify(keypoints));
        }
      }
    });

    const videoElement = videoRef.current;
    const camera = new Camera(videoElement, {
      onFrame: async () => await pose.send({ image: videoElement }),
      width: 640,
      height: 480,
    });
    camera.start();

    return () => camera.stop();
  }, [exerciseStarted]);

  const startExercise = () => {
    setExerciseStarted(true);
    fetch('http://localhost:8000/start-exercise', { method: 'POST' })
      .then((res) => res.json())
      .then((data) => console.log(data));
  };

  return (
    <div style={{ textAlign: 'center', marginTop: '20px' }}>
      <h1>Welcome to Rehab App</h1>
      {!exerciseStarted ? (
        <button onClick={startExercise}>Start Exercise</button>
      ) : (
        <>
          <div style={{ position: 'relative', width: '640px', margin: '0 auto' }}>
            <video ref={videoRef} style={{ width: '640px', height: '480px' }} />
            <canvas
              ref={canvasRef}
              width="640"
              height="480"
              style={{ position: 'absolute', top: 0, left: 0 }}
            />
          </div>
          <p style={{ fontSize: '18px', color: feedback.includes('Incorrect') ? 'red' : 'green' }}>
            {feedback}
          </p>
          <div style={{ width: '640px', margin: '20px auto' }}>
            <Bar
              data={progressData}
              options={{
                responsive: true,
                plugins: { title: { display: true, text: 'Progress Over Time' } },
              }}
            />
          </div>
        </>
      )}
    </div>
  );
}

export default App;