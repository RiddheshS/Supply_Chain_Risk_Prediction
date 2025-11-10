import React from 'react';
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Chip } from '@mui/material';

const RiskHistory = ({ data }) => {
  const getRiskLevelColor = (level) => {
    switch (level) {
      case 0: return 'success'; // Low risk
      case 1: return 'warning'; // Medium risk
      case 2: return 'error'; // High risk
      default: return 'default';
    }
  };

  const getRiskLevelText = (level) => {
    switch (level) {
      case 0: return 'Low';
      case 1: return 'Medium';
      case 2: return 'High';
      default: return 'Unknown';
    }
  };

  return (
    <TableContainer component={Paper}>
      <Table sx={{ minWidth: 650 }} aria-label="risk history table">
        <TableHead>
          <TableRow>
            <TableCell>Supplier ID</TableCell>
            <TableCell>Product ID</TableCell>
            <TableCell align="right">Quantity</TableCell>
            <TableCell align="right">Price</TableCell>
            <TableCell align="right">Delivery Time</TableCell>
            <TableCell align="right">Risk Score</TableCell>
            <TableCell>Risk Level</TableCell>
            <TableCell>Timestamp</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {data.map((row) => (
            <TableRow
              key={row.id}
              sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
            >
              <TableCell component="th" scope="row">
                {row.supplier_id}
              </TableCell>
              <TableCell>{row.product_id}</TableCell>
              <TableCell align="right">{row.quantity}</TableCell>
              <TableCell align="right">${row.price}</TableCell>
              <TableCell align="right">{row.delivery_time} days</TableCell>
              <TableCell align="right">{row.risk_score?.toFixed(4) || 'N/A'}</TableCell>
              <TableCell>
                <Chip
                  label={getRiskLevelText(Math.floor(row.risk_score * 3) || 0)}
                  color={getRiskLevelColor(Math.floor(row.risk_score * 3) || 0)}
                  size="small"
                />
              </TableCell>
              <TableCell>{new Date(row.timestamp).toLocaleString()}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default RiskHistory;
