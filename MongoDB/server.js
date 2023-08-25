const http = require('http');

const server = http.createServer((req, res) => {
  
  // Set status and headers
  res.statusCode = 200;
  res.setHeader('Content-Type', 'application/json');
  
  // Send response
  res.end('Hello World');
});

server.listen(8080);