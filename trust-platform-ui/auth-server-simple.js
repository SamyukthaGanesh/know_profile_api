// Simple Express-based auth server
const express = require('express');
const fs = require('fs');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = 3001;
const DB_FILE = path.join(__dirname, 'db.json');

// Middleware
app.use(cors());
app.use(express.json());

// Helper functions
const readDB = () => {
  const data = fs.readFileSync(DB_FILE, 'utf8');
  return JSON.parse(data);
};

const writeDB = (data) => {
  fs.writeFileSync(DB_FILE, JSON.stringify(data, null, 2));
};

const generateToken = () => {
  return Math.random().toString(36).substring(2) + Date.now().toString(36);
};

// Login endpoint
app.post('/auth/login', (req, res) => {
  const { userId, password } = req.body;
  const db = readDB();
  
  const user = db.users.find(u => u.userId === userId || u.email === userId);
  
  if (!user || password.length < 4) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }
  
  const token = generateToken();
  const session = {
    id: token,
    userId: user.id,
    createdAt: new Date().toISOString()
  };
  
  db.sessions.push(session);
  writeDB(db);
  
  const { password: _, ...userWithoutPassword } = user;
  res.status(200).json({ token, user: userWithoutPassword });
});

// Signup endpoint
app.post('/auth/signup', (req, res) => {
  const { userId, email, password, name } = req.body;
  const db = readDB();
  
  const existingUser = db.users.find(u => u.userId === userId || u.email === email);
  if (existingUser) {
    return res.status(400).json({ error: 'User already exists' });
  }
  
  const newUser = {
    id: String(Date.now()),
    userId,
    email,
    password,
    name,
    role: 'user',
    profileId: `U${1000 + db.users.length}`,
    createdAt: new Date().toISOString()
  };
  
  db.users.push(newUser);
  
  const token = generateToken();
  const session = {
    id: token,
    userId: newUser.id,
    createdAt: new Date().toISOString()
  };
  
  db.sessions.push(session);
  writeDB(db);
  
  const { password: _, ...userWithoutPassword } = newUser;
  res.status(201).json({ token, user: userWithoutPassword });
});

// Logout endpoint
app.post('/auth/logout', (req, res) => {
  const token = req.headers.authorization?.replace('Bearer ', '');
  
  if (token) {
    const db = readDB();
    db.sessions = db.sessions.filter(s => s.id !== token);
    writeDB(db);
  }
  
  res.status(200).json({ message: 'Logged out successfully' });
});

// Verify token endpoint
app.get('/auth/verify', (req, res) => {
  const token = req.headers.authorization?.replace('Bearer ', '');
  
  if (!token) {
    return res.status(401).json({ error: 'No token provided' });
  }
  
  const db = readDB();
  const session = db.sessions.find(s => s.id === token);
  
  if (!session) {
    return res.status(401).json({ error: 'Invalid token' });
  }
  
  const user = db.users.find(u => u.id === session.userId);
  
  if (!user) {
    return res.status(401).json({ error: 'User not found' });
  }
  
  const { password: _, ...userWithoutPassword } = user;
  res.status(200).json({ user: userWithoutPassword });
});

app.listen(PORT, () => {
  console.log(`🔐 Auth Server running on http://localhost:${PORT}`);
  console.log(`📝 Default credentials:`);
  console.log(`   Admin: userId="admin", password="password"`);
  console.log(`   User:  userId="user1", password="password"`);
});

