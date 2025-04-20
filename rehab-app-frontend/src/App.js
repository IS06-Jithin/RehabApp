  import React, { useRef, useEffect, useState } from 'react';
  import { Pose } from '@mediapipe/pose';
  import { Camera } from '@mediapipe/camera_utils';
  import { drawLandmarks, drawConnectors } from '@mediapipe/drawing_utils';
  import { POSE_CONNECTIONS } from '@mediapipe/pose';
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

  ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

  const EXERCISE_MAP = {
    1: "Arm abduction",
    2: "Arm VW",
    3: "Push-ups",
    4: "Leg abduction",
    5: "Leg lunge",
    6: "Squats",
  };

  function App() {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const wsRef = useRef(null);
    const noPoseHoldRef = useRef(false);
    const noPoseHoldTimeout = useRef(null);

    const [isPoseDetectedLocally, setIsPoseDetectedLocally] = useState(false);
    const [feedback, setFeedback] = useState('');
    const [suggestion, setSuggestion] = useState('');
    const [exerciseStarted, setExerciseStarted] = useState(false);
    const [progressData, setProgressData] = useState({ labels: [], datasets: [] });
    const [selectedExerciseId, setSelectedExerciseId] = useState(1);

    // Fetch historical progress data
    useEffect(() => {
      const fetchProgress = async () => {
        try {
          const response = await fetch(`http://localhost:8000/progress?exercise_id=${selectedExerciseId}`);
          const data = await response.json();
          if (Array.isArray(data) && data.length > 0) {
            const labels = data.map((entry) => new Date(entry.timestamp).toLocaleTimeString());
            const deviations = data.map((entry) => entry.deviation);
            const classifications = data.map((entry) => entry.classification);
            setProgressData({
              labels,
              datasets: [
                {
                  label: 'Average Deviation',
                  data: deviations,
                  backgroundColor: classifications.map((cls) =>
                    cls === 'Correct' ? 'rgba(75, 192, 192, 0.5)' : 'rgba(255, 99, 132, 0.5)'
                  ),
                  borderColor: classifications.map((cls) =>
                    cls === 'Correct' ? 'rgba(75, 192, 192, 1)' : 'rgba(255, 99, 132, 1)'
                  ),
                  borderWidth: 1,
                },
              ],
            });
          } else if (data.error) {
            console.error('Error fetching progress:', data.error);
            setFeedback('Failed to load progress data.');
          }
        } catch (error) {
          console.error('Fetch progress error:', error);
          setFeedback('Error fetching progress data.');
        }
      };
      fetchProgress();
    }, [selectedExerciseId]);

    // Initialize WebSocket
    useEffect(() => {
      if (!wsRef.current) {
        wsRef.current = new WebSocket('ws://localhost:8000/ws');
        wsRef.current.onopen = () => console.log('WebSocket Connected');
        wsRef.current.onerror = (error) => console.error('WebSocket Error:', error);
        wsRef.current.onclose = (event) => {
          console.log('WebSocket Disconnected:', event.code, event.reason, event.wasClean);
          setFeedback('Connection lost. Please refresh.');
          setSuggestion('');
          setExerciseStarted(false);
        };

        wsRef.current.onmessage = (event) => {
          console.log('Message from server:', event.data);
          try {
            const data = JSON.parse(event.data);
            let speakSuggestion = false;

            if (data.message && !data.feedback) {
              if (!feedback || feedback === 'Initializing...' || feedback.includes('stopped')) {
                setFeedback(data.message);
              }
              if (data.message.includes('started') || data.message.includes('stopped')) {
                setSuggestion('');
              }
            } else if (data.feedback) {
              setFeedback(data.feedback);
              if (data.feedback === 'Correct') {
                setSuggestion('');
              }
            }

            if (data.suggestion) {
              setSuggestion(data.suggestion);
              speakSuggestion = true;
            }

            if (speakSuggestion && data.suggestion) {
              const utterance = new SpeechSynthesisUtterance(data.suggestion);
              speechSynthesis.cancel();
              speechSynthesis.speak(utterance);
            }

            if (data.errors && Array.isArray(data.errors)) {
              setProgressData((prevData) => {
                const MAX_POINTS = 50;
                const newLabels = [...(prevData.labels || []), new Date().toLocaleTimeString()].slice(-MAX_POINTS);
                const currentDatasets = prevData.datasets || [];
                const avgDeviation = data.errors.reduce((sum, val) => sum + Math.abs(val), 0) / (data.errors.length || 1);
                const targetDatasetIndex = currentDatasets.findIndex((ds) => ds.label === 'Average Deviation');
                const currentData = targetDatasetIndex !== -1 ? currentDatasets[targetDatasetIndex]?.data || [] : [];
                const newData = [...currentData, avgDeviation].slice(-MAX_POINTS);
                const newDataset = {
                  label: 'Average Deviation',
                  data: newData,
                  backgroundColor: data.feedback === 'Correct' ? 'rgba(75, 192, 192, 0.5)' : 'rgba(255, 99, 132, 0.5)',
                  borderColor: data.feedback === 'Correct' ? 'rgba(75, 192, 192, 1)' : 'rgba(255, 99, 132, 1)',
                  borderWidth: 1,
                };
                const updatedDatasets = [...currentDatasets];
                if (targetDatasetIndex !== -1) {
                  updatedDatasets[targetDatasetIndex] = newDataset;
                } else {
                  updatedDatasets.push(newDataset);
                }
                return { labels: newLabels, datasets: updatedDatasets };
              });
            }
          } catch (e) {
            console.error('WS message parse/update Error:', e);
            setFeedback(`Msg Error: ${typeof event.data === 'string' ? event.data : 'Processing error.'}`);
            setSuggestion('');
          }
        };
      }

      return () => {
        if (wsRef.current) {
          if (wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.close();
          }
          wsRef.current = null;
        }
      };
    }, []);

    // Setup MediaPipe Pose
    useEffect(() => {
      if (!exerciseStarted) return;

      const currentExIdForMediaPipe = selectedExerciseId;
      console.log(`Initializing MediaPipe Pose for Exercise ID: ${currentExIdForMediaPipe}`);
      const pose = new Pose({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}` });

      pose.setOptions({
        modelComplexity: 2,
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
        enableSegmentation: false,
      });

      let keypointSequence = [];
      let frameCounter = 0;
      let camera = null;
      let isComponentMounted = true;
      let firstResultProcessed = false;

      pose.onResults((results) => {
        if (!isComponentMounted || !exerciseStarted) return;
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        if (!firstResultProcessed) {
          firstResultProcessed = true;
          setFeedback((prevFeedback) => (prevFeedback === 'Initializing...' ? '' : prevFeedback));
        }

        ctx.save();
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (results.poseLandmarks) {
          try {
            drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 4 });
            drawLandmarks(ctx, results.poseLandmarks, { color: '#FF0000', lineWidth: 2, radius: 6 });
          } catch (e) {
            console.error('Drawing Error:', e);
          }
        }

        if (results.poseWorldLandmarks) {
          setIsPoseDetectedLocally(true);
          const processFrameInterval = 5;
          if (frameCounter % processFrameInterval === 0) {
            const VISIBILITY_THRESHOLD = 0.8;
            const REQUIRED_INDICES = [11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 0, 15, 16, 19, 20, 31, 32]; // Add joint indices based on your ERR_JOINTS logic

            let allRequiredVisible = true;
            const worldKeypoints = results.poseWorldLandmarks.map((lm, index) => {
              if (REQUIRED_INDICES.includes(index) && lm.visibility < VISIBILITY_THRESHOLD) {
                allRequiredVisible = false;
              }
              return lm.visibility >= VISIBILITY_THRESHOLD ? [lm.x, lm.y, lm.z] : [0.0, 0.0, 0.0];
            });

            if (allRequiredVisible && worldKeypoints.length === 33) {
              keypointSequence.push(worldKeypoints);
              setIsPoseDetectedLocally(true);
            } else {
              setIsPoseDetectedLocally(false);
            }

            if (allRequiredVisible && worldKeypoints.length === 33) {
              keypointSequence.push(worldKeypoints);
            
              const sequenceLength = 16;
              if (keypointSequence.length >= sequenceLength) {
                const payload = {
                  label: 'keypoint_sequence',
                  keypoints: keypointSequence,
                  exercise_id: currentExIdForMediaPipe,
                };
            
                if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                  try {
                    wsRef.current.send(JSON.stringify(payload));
                  } catch (sendError) {
                    console.error('WS Send Error:', sendError);
                  }
                } else {
                  console.warn('WebSocket is not open for sending keypoints.');
                }
            
                keypointSequence = [];
              }
            } else {
              setIsPoseDetectedLocally(false); // Mark as not detected to show user-friendly message
            }
            
            const sequenceLength = 16;
            if (keypointSequence.length >= sequenceLength) {
              const payload = {
                label: 'keypoint_sequence',
                keypoints: keypointSequence,
                exercise_id: currentExIdForMediaPipe,
              };

              if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                try {
                  wsRef.current.send(JSON.stringify(payload));
                } catch (sendError) {
                  console.error('WS Send Error:', sendError);
                }
              } else {
                console.warn('WebSocket is not open for sending keypoints.');
              }
              keypointSequence = [];
            }
          }
          frameCounter++;
        } else {
          setIsPoseDetectedLocally(false);
        }
        ctx.restore();
      });

      const videoElement = videoRef.current;
      if (videoElement) {
        console.log('Setting up camera...');
        camera = new Camera(videoElement, {
          onFrame: async () => {
            if (videoElement.readyState >= 2 && pose && isComponentMounted) {
              try {
                await pose.send({ image: videoElement });
              } catch (poseError) {
                console.error('Pose send Error:', poseError);
              }
            }
          },
          width: 640,
          height: 480,
        });
        camera
          .start()
          .then(() => {
            if (isComponentMounted) console.log('Camera started.');
          })
          .catch((err) => {
            console.error('Camera start Error:', err);
            if (isComponentMounted) {
              setFeedback(`Camera Error: ${err.message}.`);
              setExerciseStarted(false);
            }
          });
      } else {
        console.error('Video element ref missing.');
        setFeedback('Video element missing.');
        setExerciseStarted(false);
      }

      return () => {
        console.log('Cleaning up MediaPipe/Camera...');
        isComponentMounted = false;
        camera?.stop();
        pose.close();
        setIsPoseDetectedLocally(false);
      };
    }, [exerciseStarted, selectedExerciseId]);

    // Manage "No pose detected" message
    useEffect(() => {
      if (!exerciseStarted) return;
    
      if (!isPoseDetectedLocally) {
        if (!noPoseHoldRef.current) {
          setFeedback('No pose detected. Please adjust your position.');
          setSuggestion('');
          noPoseHoldRef.current = true;
    
          // Start 1-second hold
          noPoseHoldTimeout.current = setTimeout(() => {
            noPoseHoldRef.current = false;
          }, 1000);
        }
      } else {
        // Only clear message if hold period has passed
        if (!noPoseHoldRef.current && feedback === 'No pose detected. Please adjust your position.') {
          setFeedback('');
        }
      }
    }, [exerciseStarted, isPoseDetectedLocally, feedback]);
    

    // Control Functions
    const handleExerciseChange = (event) => {
      setSelectedExerciseId(parseInt(event.target.value, 10));
    };

    const startExercise = () => {
      if (!selectedExerciseId) {
        alert('Please select an exercise first.');
        return;
      }
      console.log(`Starting exercise ID: ${selectedExerciseId}`);
      setFeedback('Initializing...');
      setSuggestion('');
      setProgressData({ labels: [], datasets: [] });
      setIsPoseDetectedLocally(false);
      setExerciseStarted(true);

      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        try {
          wsRef.current.send(
            JSON.stringify({
              action: 'start_exercise',
              exercise_id: selectedExerciseId,
            })
          );
        } catch (e) {
          console.error('WS Send Start Error:', e);
          setFeedback('Error starting exercise. Check connection.');
          setExerciseStarted(false);
        }
      } else {
        console.warn('WS not open on start.');
        setFeedback('Connecting... Please wait and try again.');
        setExerciseStarted(false);
      }
    };

    const stopExercise = () => {
      console.log('Stopping exercise...');
      setExerciseStarted(false);
      setFeedback('Exercise stopped.');
      setSuggestion('');
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        try {
          wsRef.current.send(JSON.stringify({ action: 'stop_exercise' }));
        } catch (e) {
          console.error('WS Send Stop Error:', e);
        }
      }
      speechSynthesis.cancel();
    };

    // JSX Rendering
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '20px', fontFamily: 'sans-serif' }}>
        <h1>AI Rehab Assistant</h1>

        {!exerciseStarted ? (
          <div style={{ textAlign: 'center', padding: '30px', border: '1px solid #ccc', borderRadius: '8px', backgroundColor: '#f9f9f9', maxWidth: '500px' }}>
            <h2>Instructions</h2>
            <p>Ensure good lighting. Stand 3-6 feet from the camera.</p>
            <div style={{ margin: '20px 0' }}>
              <label htmlFor="exerciseSelect" style={{ marginRight: '10px', fontWeight: 'bold' }}>
                Choose Exercise:
              </label>
              <select
                id="exerciseSelect"
                value={selectedExerciseId}
                onChange={handleExerciseChange}
                style={{ padding: '8px 12px', fontSize: '16px', minWidth: '200px', cursor: 'pointer' }}
              >
                {Object.entries(EXERCISE_MAP).map(([id, name]) => (
                  <option key={id} value={id}>
                    {name} (ID: {id})
                  </option>
                ))}
              </select>
            </div>
            <p style={{ color: 'red', fontWeight: 'bold', minHeight: '1.2em' }}>
              {feedback &&
              (feedback.includes('Error') ||
                feedback.includes('lost') ||
                feedback.includes('Connecting') ||
                feedback.includes('missing') ||
                feedback.includes('stopped') ||
                feedback.includes('failed'))
                ? feedback
                : ''}
            </p>
            <button
              onClick={startExercise}
              disabled={!selectedExerciseId}
              style={{
                padding: '12px 25px',
                fontSize: '18px',
                cursor: selectedExerciseId ? 'pointer' : 'not-allowed',
                marginTop: '15px',
                backgroundColor: selectedExerciseId ? '#4CAF50' : '#cccccc',
                color: 'white',
                border: 'none',
                borderRadius: '5px',
              }}
            >
              Start Exercise
            </button>
          </div>
        ) : (
          <>
            <div style={{ margin: '10px 0', fontSize: '18px', fontWeight: 'bold', textAlign: 'center', width: '640px' }}>
              Performing: <span style={{ color: '#007bff' }}>{EXERCISE_MAP[selectedExerciseId] || 'Unknown Exercise'}</span>
            </div>
            <div style={{ position: 'relative', width: '640px', height: '480px', margin: '10px auto 20px auto', border: '2px solid black', backgroundColor: '#333' }}>
              <video ref={videoRef} style={{ display: 'block', width: '100%', height: '100%', objectFit: 'cover' }} autoPlay playsInline muted />
              <canvas ref={canvasRef} width="640" height="480" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} />
            </div>
            <div
              style={{
                marginTop: '15px',
                minHeight: '3em',
                padding: '10px 15px',
                border: `2px solid ${
                  feedback.includes('Incorrect') || feedback.includes('No pose') || feedback.includes('Error') || feedback.includes('Warning')
                    ? 'red'
                    : feedback === 'Correct'
                    ? 'green'
                    : '#ccc'
                }`,
                borderRadius: '5px',
                width: '640px',
                backgroundColor: '#f8f8f8',
                textAlign: 'center',
              }}
            >
              <strong>Feedback:</strong>
              <p
                style={{
                  fontSize: '18px',
                  margin: '5px 0',
                  color:
                    feedback.includes('Incorrect') || feedback.includes('No pose') || feedback.includes('Error') || feedback.includes('Warning')
                      ? 'red'
                      : feedback === 'Correct'
                      ? '#2e7d32'
                      : '#333',
                  fontWeight: 'bold',
                  minHeight: '1.2em',
                }}
              >
                {feedback || (isPoseDetectedLocally ? 'Analyzing pose...' : 'Waiting for pose detection...')}
              </p>
            </div>
            {suggestion && (
              <div
                style={{
                  marginTop: '10px',
                  padding: '15px 20px',
                  border: '2px solid orange',
                  borderRadius: '5px',
                  width: '640px',
                  backgroundColor: '#fff3e0',
                  textAlign: 'center',
                }}
              >
                <strong style={{ fontSize: '18px', color: '#e65100' }}>Suggestion:</strong>
                <p style={{ fontSize: '24px', margin: '8px 0 0 0', color: '#e65100', fontWeight: 'bold', lineHeight: '1.3' }}>{suggestion}</p>
              </div>
            )}
            {progressData.labels && progressData.labels.length > 0 && (
              <div style={{ width: '90%', maxWidth: '640px', margin: '30px auto', padding: '15px', border: '1px solid #ddd', borderRadius: '8px', backgroundColor: '#f9f9f9' }}>
                <h2 style={{ textAlign: 'center', marginBottom: '15px' }}>Exercise Progress</h2>
                <Bar
                  data={progressData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                      title: {
                        display: true,
                        text: `Deviation for ${EXERCISE_MAP[selectedExerciseId] || 'Exercise'}`,
                        font: { size: 18 },
                      },
                      legend: {
                        display: true,
                        position: 'top',
                      },
                      tooltip: {
                        callbacks: {
                          label: (context) => `${context.dataset.label}: ${context.raw.toFixed(2)}`,
                        },
                      },
                    },
                    scales: {
                      y: {
                        beginAtZero: true,
                        title: {
                          display: true,
                          text: 'Deviation (Average Error)',
                        },
                        suggestedMax: 1.0,
                      },
                      x: {
                        title: {
                          display: true,
                          text: 'Time',
                        },
                        ticks: {
                          autoSkip: true,
                          maxTicksLimit: 10,
                        },
                      },
                    },
                    animation: {
                      duration: 500,
                    },
                  }}
                />
              </div>
            )}
            <button
              onClick={stopExercise}
              style={{ padding: '12px 25px', fontSize: '18px', cursor: 'pointer', marginTop: '20px', backgroundColor: '#f44336', color: 'white', border: 'none', borderRadius: '5px' }}
            >
              Stop Exercise
            </button>
          </>
        )}
      </div>
    );
  }

  export default App;