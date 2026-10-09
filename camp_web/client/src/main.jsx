import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import Landing from './views/Landing.jsx';
import './index.css';

// Tiny dependency-free hash router: #/app shows the camp app, anything else the landing hero.
function Root() {
  const [route, setRoute] = useState(window.location.hash);
  useEffect(() => {
    const onHash = () => setRoute(window.location.hash);
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  const isApp = route.startsWith('#/app');
  return isApp ? <App /> : <Landing />;
}

createRoot(document.getElementById('root')).render(<Root />);
