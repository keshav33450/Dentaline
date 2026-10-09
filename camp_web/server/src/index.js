import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import mongoose from 'mongoose';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import api from './routes/api.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 8080);
const MONGO = process.env.MONGODB_URI || 'mongodb://127.0.0.1:27017/dentalx_camp';

const app = express();
app.use(cors());
app.use(express.json({ limit: '25mb' }));
app.use('/api', api);

// serve built React app in production (client/dist)
const clientDist = path.resolve(__dirname, '../../client/dist');
app.use(express.static(clientDist));
app.get('*', (req, res, next) => {
  if (req.path.startsWith('/api')) return next();
  res.sendFile(path.join(clientDist, 'index.html'), (err) => { if (err) next(); });
});

async function start() {
  try {
    await mongoose.connect(MONGO);
    console.log('MongoDB connected:', MONGO.replace(/\/\/([^@]*@)?/, '//'));
  } catch (e) {
    console.error('MongoDB connection failed:', e.message);
    console.error('Set MONGODB_URI (local mongod or Atlas). Server will still start but DB ops will fail.');
  }
  app.listen(PORT, '127.0.0.1', () => console.log(`DentalX Camp API on http://127.0.0.1:${PORT}`));
}
start();
