import React from 'react';
import { useCurrentFrame, useVideoConfig, AbsoluteFill } from 'remotion';

export interface ComponentProps {
  style?: React.CSSProperties;
  animations?: any[];
  transitions?: any[];
}

export const Background: React.FC<{ bgType: string } & ComponentProps> = ({ bgType, style }) => {
  const bgColors: Record<string, string> = {
    dark_radial: 'radial-gradient(circle, #1b263b 0%, #0d1b2a 100%)',
    blueprint_grid: '#1a5276',
    cream_pop: '#fdf6e3',
    near_black: '#0a0f1d',
    warm_paper: '#f4ede4',
    neon_vignette: 'radial-gradient(circle at center, #1b0c2a 0%, #08030d 100%)',
  };
  const background = bgColors[bgType] || bgColors.dark_radial;
  return (
    <AbsoluteFill style={{ background, ...style }} />
  );
};

export const Chart: React.FC<{ chartType: string } & ComponentProps> = ({ chartType, style }) => {
  return (
    <div style={{
      border: '2px dashed #415a77',
      borderRadius: '16px',
      backgroundColor: 'rgba(27, 38, 59, 0.4)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#778da9',
      fontFamily: 'system-ui, sans-serif',
      fontWeight: 'bold',
      ...style
    }}>
      [Chart: {chartType}]
    </div>
  );
};

export const Image: React.FC<{ src: string } & ComponentProps> = ({ src, style }) => {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', ...style }}>
      <img src={src} style={{ width: '100%', height: '100%', objectFit: 'contain' }} alt="Graphic" />
    </div>
  );
};

export const Subtitle: React.FC<{ words: string[]; timings: number[][]; emphasisColor?: string; captionStyle?: string } & ComponentProps> = ({ words, timings, emphasisColor = '#fca311', captionStyle, style }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const currentTime = frame / fps;

  // Find the currently active word index based on currentTime
  let activeIndex = -1;
  for (let i = 0; i < timings.length; i++) {
    if (i < timings.length) {
      const [start, end] = timings[i];
      if (currentTime >= start && currentTime < end) {
        activeIndex = i;
        break;
      }
    }
  }

  return (
    <div style={{
      display: 'flex',
      flexWrap: 'wrap',
      justifyContent: 'center',
      alignItems: 'center',
      width: '100%',
      textAlign: 'center',
      lineHeight: '1.4',
      padding: '20px',
      ...style
    }}>
      {words.map((w, idx) => (
        <span
          key={idx}
          style={{
            color: idx === activeIndex ? emphasisColor : '#ffffff',
            textShadow: idx === activeIndex
              ? `0 0 20px ${emphasisColor}, 0 4px 10px rgba(0,0,0,0.5)`
              : '0 4px 10px rgba(0,0,0,0.5)',
            marginRight: '12px',
            transform: idx === activeIndex ? 'scale(1.15)' : 'scale(1.0)',
            transition: 'transform 0.1s ease-out, color 0.1s ease-out',
            display: 'inline-block',
            textTransform: captionStyle?.toLowerCase().includes("uppercase") ? "uppercase" : "none",
          }}
        >
          {w}
        </span>
      ))}
    </div>
  );
};

export const Video: React.FC<{ src: string } & ComponentProps> = ({ src, style }) => {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', ...style }}>
      [Video: {src}]
    </div>
  );
};

export const Transition: React.FC<ComponentProps> = () => null;
export const Animation: React.FC<ComponentProps> = () => null;
