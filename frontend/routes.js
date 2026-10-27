
module.exports = (app, datapacks) => {

	Object.assign(app.locals, {
		basePath: "/"
	})

	// console.log(app.locals)

	app.get('/', function(req, res) {

		res.locals = {
			title: `Chat Visualizer`,
			styles: ["css/roboto.css", "css/main.css", "css/index.css"],
			scripts: ["js/index.js"],
			datapacks
		}

		res.render('index');
	});

	app.get('/visualizations/', function (req, res) {
		res.redirect('/')
	})

	app.get('/visualizations/:datapackID/', function (req, res) {
		const datapackID = req.params.datapackID
		const datapack = datapacks[datapackID]
		if (!datapack) {
			Object.assign(res.locals, {
				datapackID: datapackID,
				styles: ["css/roboto.css", "css/main.css", "css/chat.css"],
			})
			return res.status(404).render("404datapack")
		}
		// console.log("datapack.chatsByClassName", datapack.chatsByClassName)

		showFullList = "full" in req.query ||
			Object.keys(datapack["chats"]).length < 500

		res.locals = {
			title: `${datapackID} Datapack`,
			styles: ["css/roboto.css", "css/main.css", "css/chat_index.css"],
			scripts: ["js/chat_index.js"],
			// should possibly be sorted somehow
			classNames: Object.keys(datapack.chatsByClassName),
			showFullList: showFullList
		}

		res.locals = Object.assign(res.locals, datapack)

		if (!showFullList)
			res.locals.chatsByClassName = res.locals.chatsByClassNameLimited

		res.render("chat_index")
	})

	app.get('/visualizations/:datapackID/chat/:chatID/', function (req, res) {
		const datapackID = req.params.datapackID;
		const datapack = datapacks[datapackID];
		if (!datapack) {
			Object.assign(res.locals, {
				datapackID: datapackID,
				styles: ["css/roboto.css", "css/main.css", "css/chat.css"],
			})
			return res.status(404).render("404datapack")
		}

		const chatID = req.params.chatID;
		const chat = datapack.chats[chatID]
		res.locals = {
			title: `${chatID} visualization`,
			styles: ["css/roboto.css", "css/main.css", "css/chat.css"],
			scripts: ["vendor/Chart.min.js", "vendor/patternomaly.min.js", "js/chat.js"],
			"datapackID": req.params.datapackID,
			"chatID": req.params.chatID,
			options: require("./options.json"),
			chat
		}
		if (!chat) {
			const otherDatapackIDs = Object.keys(datapacks).filter(
				locDatapackID => chatID in datapacks[locDatapackID].chats &&
				                 locDatapackID !== datapackID)
			res.locals.otherDatapackIDs = otherDatapackIDs.length ?
				otherDatapackIDs : null
			return res.status(404).render("404chat")
		}
		res.render("chat")
	})

}
