import { useState } from "react";
import StoreImage from "./StoreImage";
import Nav from "../Navigation/Nav";
import "./Home.css"
export default function Home() {
    return (
        <div>
            <Nav/>
            <div class="page-content">
            <h1 class="page-title">Visual Question and Answering System</h1>
            <p class="page-description">The web page can help users to respond to the queries related to the image by effectively utilizing 
             the textual information available in the images to offer additional useful cues and improve understanding of the visual elements in the content. 
             This system can precisely provide answers by properly interpreting the image features.</p>
            </div>
            <StoreImage/>
        </div>
    );
}
