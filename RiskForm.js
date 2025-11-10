import React, { useState } from 'react';
import { TextField, Button, Box, Grid } from '@mui/material';

const RiskForm = ({ onSubmit }) => {
  const [formData, setFormData] = useState({
    supplier_id: '',
    product_id: '',
    quantity: '',
    price: '',
    delivery_time: '',
    supplier_reliability: '',
    demand_volatility: '',
    geopolitical_risk: '',
    economic_indicators: '',
    historical_delays: ''
  });

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Convert string values to numbers where appropriate
    const processedData = {
      ...formData,
      quantity: parseFloat(formData.quantity),
      price: parseFloat(formData.price),
      delivery_time: parseFloat(formData.delivery_time),
      supplier_reliability: parseFloat(formData.supplier_reliability),
      demand_volatility: parseFloat(formData.demand_volatility),
      geopolitical_risk: parseFloat(formData.geopolitical_risk),
      economic_indicators: parseFloat(formData.economic_indicators),
      historical_delays: parseFloat(formData.historical_delays)
    };
    onSubmit(processedData);
  };

  return (
    <Box component="form" onSubmit={handleSubmit} sx={{ mt: 1 }}>
      <Grid container spacing={2}>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Supplier ID"
            name="supplier_id"
            value={formData.supplier_id}
            onChange={handleChange}
            required
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Product ID"
            name="product_id"
            value={formData.product_id}
            onChange={handleChange}
            required
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Quantity"
            name="quantity"
            type="number"
            value={formData.quantity}
            onChange={handleChange}
            required
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Price"
            name="price"
            type="number"
            value={formData.price}
            onChange={handleChange}
            required
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Delivery Time (days)"
            name="delivery_time"
            type="number"
            value={formData.delivery_time}
            onChange={handleChange}
            required
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Supplier Reliability (0-1)"
            name="supplier_reliability"
            type="number"
            inputProps={{ min: 0, max: 1, step: 0.01 }}
            value={formData.supplier_reliability}
            onChange={handleChange}
            required
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Demand Volatility (0-1)"
            name="demand_volatility"
            type="number"
            inputProps={{ min: 0, max: 1, step: 0.01 }}
            value={formData.demand_volatility}
            onChange={handleChange}
            required
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Geopolitical Risk (0-1)"
            name="geopolitical_risk"
            type="number"
            inputProps={{ min: 0, max: 1, step: 0.01 }}
            value={formData.geopolitical_risk}
            onChange={handleChange}
            required
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Economic Indicators (0-1)"
            name="economic_indicators"
            type="number"
            inputProps={{ min: 0, max: 1, step: 0.01 }}
            value={formData.economic_indicators}
            onChange={handleChange}
            required
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Historical Delays (0-1)"
            name="historical_delays"
            type="number"
            inputProps={{ min: 0, max: 1, step: 0.01 }}
            value={formData.historical_delays}
            onChange={handleChange}
            required
          />
        </Grid>
        <Grid item xs={12}>
          <Button type="submit" fullWidth variant="contained" sx={{ mt: 3, mb: 2 }}>
            Predict Risk
          </Button>
        </Grid>
      </Grid>
    </Box>
  );
};

export default RiskForm;
