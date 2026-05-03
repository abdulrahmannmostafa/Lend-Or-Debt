import React from 'react';
import { introStyles } from '../styles';

const Intro = () => {
  return (
    <section style={introStyles.section}>
      <div style={introStyles.container}>
        <h1 style={introStyles.title}>Taiwanese Credit Default Risk Prediction</h1>
        <img src="/img1.jpg" alt="Credit Default Risk Analysis" style={introStyles.image} />
        <p style={introStyles.paragraph}>
          Our machine learning model predicts whether a Taiwanese credit cardholder 
          will default on payment in the next month, providing real-time risk assessment 
          and comprehensive financial behavior analysis.
        </p>
      </div>
    </section>
  );
};

export default Intro;
