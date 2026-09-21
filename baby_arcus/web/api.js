export async function get(path){const response=await fetch(path,{cache:"no-store"});if(!response.ok)throw new Error("Service response "+response.status);return response.json();}

