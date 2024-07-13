const fs = require("fs")
const path = require("path")

function limitObj(obj, amount) {
	const limitedKeys = Object.keys(obj).slice(0, amount)
	const ret = {}
	for (key of limitedKeys) ret[key] = obj[key]
	return ret
}


module.exports = function getDatapacks(datapackPaths) {
	const startTime = +new Date();
	console.log("datapackPaths", datapackPaths)
	let datapackFiles = datapackPaths.reduce((files, dir) =>
		files.concat(fs.readdirSync(dir).map(file => path.join(dir,file))), [])
	datapackFiles = [...new Set(datapackFiles.filter(path => path.endsWith(".json")))]
	console.log("datapackFiles", datapackFiles)

	const datapacks = {}
	datapackFiles.forEach((datapackFile, i) => {
		console.log(`reading file ${i+1}/${datapackFiles.length}`)

		let file, json;
		// console.log("path:", path.join(datapackPath, datapackFile))
		try { file = fs.readFileSync(path.join(datapackFile), 'utf8')	}
		catch(err) { throw new Error("could not read file " + datapackFile) }
		try { json = JSON.parse(file) }
		catch(err) { throw new Error("could not parse JSON file " + datapackFile) }

		if ("datapackID" in json) { // if it is actually a datapack
			datapacks[json["datapackID"]] = json;
		} else {
			console.log(datapackFile, "does not seem to be a datapack")
		}

	})

	console.log("datapacks", Object.keys(datapacks))

	// create additional data for easy templating
	for (let datapackID in datapacks) {
		datapacks[datapackID].confusionMatrix = {}

		// annotate messages
		for (const chatID in datapacks[datapackID].chats) {
			const chat = datapacks[datapackID].chats[chatID]
			chat.content = chat.content.map(c =>
				c == null ? { type: "message" } : c)
			if (!chat.authorOnRightSide) {
				// by default, the first message is on the right
				const firstMessage = chat.content.find(ct => ct.type === "message")
				chat.authorOnRightSide = firstMessage ? firstMessage.author : undefined
			}

			// add message/separator number
			let msgNo = 0;
			let sepNo = 0;
			for (let ct of chat.content) {
				if (ct == null) ct = { type: "message" }
				switch(ct.type) {
					case "message":
					ct.number = ++msgNo;
					ct.isMessage = true;
					if (ct.labels && ct.labels.length) ct.hasLabels = true

					ct.alignment = ct.author === chat.authorOnRightSide ?
						"right" : "left"

					// the empty string is a valid message
					ct.isEmpty = !ct.body && ct.body !== "" // null or undefined
					if (!ct.time) ct.hasMissingDate = true;
					if (!ct.author) ct.hasMissingAuthor = true;
					break;

					case "separator":
					ct.number = ++sepNo;
					ct.isSeparator = true;
					break;
				}
			}
			chat.noOfMessages = msgNo;
			chat.noOfSeparators = sepNo;

			const firstWarningMsg = chat.content.find(ct => (ct.HTMLclassNames || []).includes("warning"));
			chat.firstWarningMsgNo = firstWarningMsg ?
				firstWarningMsg.number : undefined
			chat.firstWarningMsgPercent = chat.firstWarningMsgNo ?
				Math.round(chat.firstWarningMsgNo/chat.noOfMessages*100*100)/100 :
				undefined

			chat.otherDatapackIDs = Object.keys(datapacks).filter(
				locDatapackID => chatID in datapacks[locDatapackID].chats &&
				                 locDatapackID !== datapackID
			)
			chat.otherDatapackIDs = chat.otherDatapackIDs.length ? chat.otherDatapackIDs : null

			// add true/false prediction annotation
			if (chat.className && chat.predictedClassName) {
				chat.truePrediction = chat.predictedClassName === chat.className
				chat.falsePrediction = chat.predictedClassName !== chat.className

				// update confusion matrix (cm)
				const cm = datapacks[datapackID].confusionMatrix
				if (!(chat.className in cm))
					cm[chat.className] = {}
				if (!(chat.predictedClassName in cm[chat.className]))
					cm[chat.className][chat.predictedClassName] = 0
				cm[chat.className][chat.predictedClassName] += 1
				datapacks[datapackID].confusionMatrix = cm

			}

		}

		// create map of chats by class name
		chatsByClassName = {}
		for (const chatID in datapacks[datapackID].chats) {
			const chat = datapacks[datapackID].chats[chatID]
			const className = chat.className
			if (!(className in chatsByClassName)) chatsByClassName[className] = {}
			chatsByClassName[className][chatID] = chat
		}
		datapacks[datapackID].chatsByClassName = chatsByClassName


		const amount = 500
		const classNames = Object.keys(chatsByClassName).sort()
		datapacks[datapackID].chatsByClassNameLimited = {}
		for (className of classNames) {
			datapacks[datapackID].chatsByClassNameLimited[className] =
				limitObj(chatsByClassName[className], amount)
		}

		// console.log("datapacks[datapackID].confusionMatrix = ",datapacks[datapackID].confusionMatrix)
	}

	console.log("parsed all datapacks:", Object.keys(datapacks))
	console.log(`Finished at ${new Date()}. Took ${((new Date() - startTime)/1000).toFixed(2)}s`)
	return datapacks
}
