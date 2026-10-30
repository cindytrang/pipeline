const express = require('express');
const path = require('node:path');
const hbs = require('hbs');
const app = express();

// register custom partials and helpers
hbs.registerPartials(__dirname + '/views/partials');
const handlebarsHelpers = require('./utils/handlebarsHelpers');
handlebarsHelpers(hbs)

// set the view engine to use handlebars
app.set('view engine', 'hbs');
app.set('views', __dirname + '/views');

app.use(express.static(__dirname + '/src'));
app.use('/vendor', express.static(__dirname + '/vendor'));
app.use(express.static(path.join(__dirname, 'public')));

// get chat datapacks
const getDatapacks = require("./utils/getDatapacks")
const datapackPaths = [process.env.DATAPACK_DIR ? path.resolve(process.env.DATAPACK_DIR) : path.join(__dirname, "datapacks")]

const datapacks = getDatapacks(datapackPaths)

// set up routes
require("./routes")(app, datapacks)


// Keep the demo local by default. Importing the app for tests does not open a port.
if (require.main === module) {
  const port = Number(process.env.PORT || 3000);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('PORT must be between 1 and 65535');
  const host = process.env.HOST || '127.0.0.1';
  app.listen(port, host, () => console.log(`Chat demo: http://${host}:${port}`));
}
module.exports = app;
