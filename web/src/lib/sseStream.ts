/** Read an SSE response and hand each `data:` payload to `onData`. */
export async function readSseData(
  res: Response,
  onData: (payload: string) => void,
): Promise<void> {
  if (!res.body) throw new Error("No response body");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const chunk = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const dataLine = chunk.split("\n").find((l) => l.startsWith("data:"));
      if (dataLine) onData(dataLine.slice(5).trim());
    }
  }
}
