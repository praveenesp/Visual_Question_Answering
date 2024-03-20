import { useState } from "react";
import StoreImage from "./StoreImage";
import Nav from "../Navigation/Nav";
export default function Home() {
    return (
        <div>
            <Nav/>
            <StoreImage/>
        </div>
    );
}
