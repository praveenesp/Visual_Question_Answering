import React, { useState } from 'react';
import Home from './components/Home/Home';
import Login from './components/googleSingin/Login';
import Signup from './components/googleSingin/Singup';
import { Routes, Route } from 'react-router-dom';
function App()
{
  return (
    <div>
       {/* <Signin/> */}
            <Routes>
               <Route path="/" element={<Signup />} />
                <Route path="/login" element={<Login />} />
                <Route path="/home" element={<Home/>}/>
            </Routes>
    </div>
  );
}

export default App;
