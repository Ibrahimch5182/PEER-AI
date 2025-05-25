// src/firebase.js
import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyCl6fJEEcZKLTj1VWTCziy9GsNQrf1ddtc",
  authDomain: "peer-ai-auth.firebaseapp.com",
  projectId: "peer-ai-auth",
  storageBucket: "peer-ai-auth.firebasestorage.app",
  messagingSenderId: "409138984069",
  appId: "1:409138984069:web:988545005008cc6bb565ec",
  measurementId: "G-4Y5P7SHLGB"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();
import { setPersistence, browserLocalPersistence } from "firebase/auth";

setPersistence(auth, browserLocalPersistence);
