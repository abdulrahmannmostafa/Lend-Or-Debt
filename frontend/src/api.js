// Base API URL — set VITE_API_URL in your Vercel environment variables
// e.g. https://your-backend.up.railway.app/api
export const API = import.meta.env.VITE_API_URL || "http://localhost:5000/api";

export async function post(url, body) {
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return await res.json();
  } catch (e) {
    return { error: e.message };
  }
}