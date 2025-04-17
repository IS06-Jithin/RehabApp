import React, { useRef, useEffect, useState } from 'react';
import { Pose } from '@mediapipe/pose';
import { Camera } from '@mediapipe/camera_utils';
// *** ENSURE THESE IMPORTS ARE CORRECT ***
import { drawLandmarks, drawConnectors } from '@mediapipe/drawing_utils';
import { POSE_CONNECTIONS } from '@mediapipe/pose';
// ****************************************
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const [isPoseDetectedLocally, setIsPoseDetectedLocally] = useState(false);
  const [feedback, setFeedback] = useState('');
  const [exerciseStarted, setExerciseStarted] = useState(false);
  const [progressData, setProgressData] = useState({ labels: [], datasets: [] });

  // Initialize WebSocket connection - Runs ONCE on mount
  useEffect(() => {
    // (WebSocket setup code remains the same as previous correct version)
    if (!wsRef.current) {
      wsRef.current = new WebSocket('ws://localhost:8000/ws');
      wsRef.current.onopen = () => console.log('WebSocket Connected');
      wsRef.current.onerror = (error) => console.error('WebSocket Error:', error);
      wsRef.current.onclose = (event) => {
          console.log('WebSocket Disconnected:', event.code, event.reason, event.wasClean);
      };
      wsRef.current.onmessage = (event) => {
        console.log('Message from server:', event.data);
        try {
          const data = JSON.parse(event.data);
          if (data.message) {
            setFeedback(data.message);
            const utterance = new SpeechSynthesisUtterance(data.message);
            speechSynthesis.speak(utterance);
          }
          if (data.deviations) {
            setProgressData((prevData) => {
              const MAX_POINTS = 50;
              const newLabels = [...prevData.labels, new Date().toLocaleTimeString()].slice(-MAX_POINTS);
              const currentDatasets = prevData.datasets || [];
              const targetDatasetIndex = currentDatasets.findIndex(ds => ds.label === 'Torso Deviation');
              const currentData = targetDatasetIndex !== -1 ? currentDatasets[targetDatasetIndex]?.data || [] : [];
              const newData = [...currentData, data.deviations.torso || 0].slice(-MAX_POINTS);
              const newDataset = { label: 'Torso Deviation', data: newData, backgroundColor: 'rgba(75, 192, 192, 0.5)'};
              const updatedDatasets = [...currentDatasets];
              if (targetDatasetIndex !== -1) { updatedDatasets[targetDatasetIndex] = newDataset; }
              else { updatedDatasets.push(newDataset); }
              return { labels: newLabels, datasets: updatedDatasets };
            });
          }
        } catch (e) {
          console.error('WS message parse/update Error:', e);
          setFeedback(typeof event.data === 'string' ? `Msg Error: ${event.data}` : 'Msg processing error.');
        }
      };
    }
    return () => {
      if (wsRef.current) {
        if (wsRef.current.readyState === WebSocket.OPEN) wsRef.current.close();
        wsRef.current = null;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Setup MediaPipe Pose - Runs when exerciseStarted changes
  useEffect(() => {
    if (!exerciseStarted) return;
    console.log('Initializing MediaPipe Pose...');
    const pose = new Pose({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}` });
    pose.setOptions({ modelComplexity: 1, smoothLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5, enableSegmentation: false });

    let keypointSequence = [];
    let frameCounter = 0;
    let camera = null;
    let isComponentMounted = true;
    let firstResultProcessed = false;

    pose.onResults((results) => {
        if (!isComponentMounted || !exerciseStarted) return;
        const canvas = canvasRef.current; if (!canvas) { return; }
        const ctx = canvas.getContext('2d'); if (!ctx) { return; }

        if (!firstResultProcessed) {
            firstResultProcessed = true;
            setFeedback(prevFeedback => prevFeedback === 'Initializing...' ? '' : prevFeedback);
        }

        ctx.save(); ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (results.poseLandmarks) {
             // *** REFACTORED: Unconditional set based on result ***
             setIsPoseDetectedLocally(true);
             // *** END REFACTOR ***

             try {
                  drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 4 });
                  drawLandmarks(ctx, results.poseLandmarks, { color: '#FF0000', lineWidth: 2, radius: 6 });
             } catch(e) { console.error("Drawing Error:", e); }

             const processFrameInterval = 5;
             if (frameCounter % processFrameInterval === 0) {
                 const keypoints = results.poseLandmarks.map(lm => [lm.x, lm.y, lm.z, lm.visibility || 0]);
                 keypointSequence.push(keypoints);
                 const sequenceLength = 10;
                 if (keypointSequence.length >= sequenceLength) {
                    
                    // Create a payload object that includes a label if needed.
                    const payload = {
                      label: 'keypoint_sequence', // Change or remove if you don't have a specific label.
                      keypoints: keypointSequence
                    };

                    // Log the payload before sending it.
                    console.log("Sending payload to server:", payload);

                    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                      wsRef.current.send(JSON.stringify(payload));
                    } else {
                      console.warn('WS not open for send.');
                    }

                     if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                         wsRef.current.send(JSON.stringify(keypointSequence));
                     } else { console.warn('WS not open for send.'); }
                     keypointSequence = [];
                 }
             }
             frameCounter++;
        } else {
             // *** REFACTORED: Unconditional set based on result ***
             setIsPoseDetectedLocally(false);
             // *** END REFACTOR ***
        }
        ctx.restore();
    });

    const videoElement = videoRef.current;
    if (videoElement) {
      console.log("Setting up camera...");
      camera = new Camera(videoElement, {
        onFrame: async () => {
            if (videoElement.readyState >= 2 && pose && isComponentMounted) {
                try { await pose.send({ image: videoElement }); }
                catch (poseError) { console.error("Pose send Error:", poseError); }
            }
        }, width: 640, height: 480,
      });
      camera.start().then(() => {
        if (isComponentMounted) console.log("Camera started.");
      }).catch(err => {
        console.error("Camera start Error:", err);
        if (isComponentMounted) { setFeedback(`Camera Error: ${err.message}.`); setExerciseStarted(false); }
      });
    } else {
      console.error('Video element ref missing.');
      setFeedback('Video element missing.'); setExerciseStarted(false);
    }

    // Cleanup for THIS MediaPipe/Camera effect
    return () => {
      console.log('Cleaning up MediaPipe/Camera...');
      isComponentMounted = false;
      camera?.stop();
      pose.close();
      // Setter call in cleanup is OK and doesn't require adding setter to deps
      setIsPoseDetectedLocally(false);
    };
  // This effect correctly depends ONLY on exerciseStarted after the refactoring above
  }, [exerciseStarted]);


  // Effect for managing user feedback based on detection status
  // This is the SECOND useEffect, its dependencies are correct.
  useEffect(() => {
    if (!exerciseStarted) return;
    const showNoPoseMessage = feedback !== 'Initializing...' && !isPoseDetectedLocally && !feedback;
    const clearNoPoseMessage = isPoseDetectedLocally && feedback === 'No pose detected. Please adjust your position.';

    if (showNoPoseMessage) {
      setFeedback('No pose detected. Please adjust your position.');
    } else if (clearNoPoseMessage) {
      setFeedback('');
    }
  // This hook correctly depends on these three values
  }, [exerciseStarted, isPoseDetectedLocally, feedback]);


  // --- Control Functions (remain the same) ---
  const startExercise = () => {
    console.log('Starting exercise...');
    setFeedback('Initializing...');
    setProgressData({ labels: [], datasets: [] });
    setIsPoseDetectedLocally(false);
    setExerciseStarted(true);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try { wsRef.current.send(JSON.stringify({ action: 'start_exercise' })); }
      catch (e) { console.error("WS Send Start Error:", e); }
    } else { console.warn('WS not open on start.'); setFeedback("Connecting..."); }
  };

  const stopExercise = () => {
    console.log('Stopping exercise...');
    setExerciseStarted(false);
    setFeedback('Exercise stopped.');
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
       try { wsRef.current.send(JSON.stringify({ action: 'stop_exercise' })); }
       catch (e) { console.error("WS Send Stop Error:", e); }
    }
  };

  // --- JSX Rendering (remains the same) ---
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '20px', fontFamily: 'sans-serif' }}>
      <h1>AI Rehab Assistant</h1>
      {!exerciseStarted ? (
        <div style={{ textAlign: 'center', padding: '30px', border: '1px solid #ccc', borderRadius: '8px', backgroundColor: '#f9f9f9', maxWidth: '500px' }}>
          <h2>Instructions</h2>
          <p>Ensure good lighting. Stand 3-6 feet from camera.</p>
          <p style={{color: 'red', fontWeight: 'bold', minHeight: '1.2em'}}>
            {feedback && (feedback.includes('Error') || feedback.includes('lost') || feedback.includes('Connecting') || feedback.includes('missing')) ? feedback : ''}
          </p>
          <button onClick={startExercise} style={{ padding: '12px 25px', fontSize: '18px', cursor: 'pointer', marginTop: '15px', backgroundColor: '#4CAF50', color: 'white', border: 'none', borderRadius: '5px' }}>
            Start Exercise
          </button>
        </div>
      ) : (
        <>
          <div style={{ position: 'relative', width: '640px', height: '480px', margin: '20px auto', border: '2px solid black', backgroundColor: '#333' }}>
            <video ref={videoRef} style={{ display: 'block', width: '100%', height: '100%', objectFit: 'cover' }} autoPlay playsInline muted />
            <canvas ref={canvasRef} width="640" height="480" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} />
          </div>
          <div style={{ marginTop: '15px', minHeight: '3em', padding: '10px 15px', border: `2px solid ${feedback.includes('Incorrect') || feedback.includes('No pose') || feedback.includes('Error') || feedback.includes('Warning') ? 'red' : 'green'}`, borderRadius: '5px', width: '640px', backgroundColor: '#f8f8f8', textAlign: 'center' }}>
            <strong>Feedback:</strong>
            <p style={{ fontSize: '18px', margin: '5px 0', color: feedback.includes('Incorrect') || feedback.includes('No pose') || feedback.includes('Error') || feedback.includes('Warning') ? 'red' : '#2e7d32', fontWeight: 'bold', minHeight: '1.2em' }}>
              {feedback || (isPoseDetectedLocally ? 'Analyzing pose...' : 'Waiting for pose detection...')}
            </p>
          </div>
          {progressData.labels && progressData.labels.length > 0 && (
            <div style={{ width: '90%', maxWidth: '640px', margin: '30px auto', padding: '10px', border: '1px solid #ddd', borderRadius: '5px' }}>
              <h2>Progress Over Time</h2>
              <Bar data={progressData} options={{ responsive: true, maintainAspectRatio: true, plugins: { title: { display: true, text: 'Deviation Trends' }, legend: { display: true, position: 'top'} }, scales: { y: { beginAtZero: true, title: { display: true, text: 'Deviation Magnitude' } }, x: { title: { display: true, text: 'Time' }, ticks: { autoSkip: true, maxTicksLimit: 10 } } }, animation: { duration: 0 } }} />
            </div>
          )}
          <button onClick={stopExercise} style={{ padding: '12px 25px', fontSize: '18px', cursor: 'pointer', marginTop: '20px', backgroundColor: '#f44336', color: 'white', border: 'none', borderRadius: '5px' }}>
            Stop Exercise
          </button>
        </>
      )}
    </div>
  );
}

export default App;