import React, { useState, useEffect } from 'react';
import { Container, Typography, Box, Paper, Grid, TextField, Button, Alert } from '@mui/material';
import { Line } from 'react-chartjs-2';
import axios from 'axios';
import RiskForm from './components/RiskForm';
import RiskHistory from './components/RiskHistory';

function App() {
  const [riskHistory, setRiskHistory] = useState([]);
  const [alert, setAlert] = useState(null);

  useEffect(() => {
    fetchRiskHistory();
  }, []);

  const fetchRiskHistory = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/risk_history');
      setRiskHistory(response.data);
    } catch (error) {
      console.error('Error fetching risk history:', error);
    }
  };

  const handleRiskPrediction = async (formData) => {
    try {
      const response = await axios.post('http://localhost:5000/api/predict_risk', formData);
      setAlert({ type: 'success', message: `Risk prediction completed. Risk Level: ${response.data.risk_level}, Score: ${response.data.risk_score.toFixed(4)}` });
      fetchRiskHistory(); // Refresh history
    } catch (error) {
      setAlert({ type: 'error', message: 'Error predicting risk. Please try again.' });
      console.error('Error predicting risk:', error);
    }
  };

  const chartData = {
    labels: riskHistory.map(item => new Date(item.timestamp).toLocaleDateString()),
    datasets: [{
      label: 'Risk Score',
      data: riskHistory.map(item => item.risk_score),
      borderColor: 'rgb(75, 192, 192)',
      tension: 0.1
    }]
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom align="center">
          Supply Chain Risk Prediction Dashboard
        </Typography>

        {alert && (
          <Alert severity={alert.type} sx={{ mb: 2 }} onClose={() => setAlert(null)}>
            {alert.message}
          </Alert>
        )}

        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h5" gutterBottom>
                Risk Prediction Form
              </Typography>
              <RiskForm onSubmit={handleRiskPrediction} />
            </Paper>
          </Grid>

          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h5" gutterBottom>
                Risk Score Trend
              </Typography>
              <Line data={chartData} />
            </Paper>
          </Grid>

          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h5" gutterBottom>
                Risk History
              </Typography>
              <RiskHistory data={riskHistory} />
            </Paper>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
}

export default App;
