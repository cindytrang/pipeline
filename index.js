const express = require('express');
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
app.use(express.static('public'));

// get chat datapacks
const getDatapacks = require("./utils/getDatapacks")
const datapackPaths = require("./nodemon.json")["watch"].filter(p => p !== ".")

const datapacks = getDatapacks(datapackPaths)

// set up routes
require("./routes")(app, datapacks)


console.log("listening on localhost:3000")
app.listen(3000);

