// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import {getAuth,GoogleAuthProvider} from "firebase/auth";
import {getStorage} from "firebase/storage";
import {getFirestore} from "firebase/firestore";
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "visual-question-answerin-4c71c.firebaseapp.com",
  projectId: "visual-question-answerin-4c71c",
  storageBucket: "visual-question-answerin-4c71c.appspot.com",
  messagingSenderId: "384502795515",
  appId: "1:384502795515:web:2739dacea929b3b36d745e",
  measurementId: "G-FXJ45993ZG"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth= getAuth(app);
const provider=new GoogleAuthProvider();
const imgDB=getStorage(app);
const txtDB=getFirestore(app);
export {auth,provider,imgDB,txtDB};
