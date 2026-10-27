
module.exports = function(hbs) {
	hbs.registerHelper('size', function (arr) {
	  return Array.isArray(arr) ? arr.length :
	         typeof arr === "object" ? Object.keys(arr).length :
	         undefined
	});
	hbs.registerHelper('join', function (arr, options) {
	  return arr ?
	  	arr.join(options.hash.delimiter || ", ") :
	  	"";
	});
		// formats like HH:MM (UTC)
	hbs.registerHelper('dateTimeHumanLocal', function(timestamp) {
		const date = new Date(timestamp).toDateString()
		const time = new Date(timestamp).toLocaleTimeString()
		return  `${date}, ${time}`
	});
	hbs.registerHelper('timeUTC', function(timestamp) {
		return new Date(timestamp).toISOString()
	});
	hbs.registerHelper('timeHumanUTC', function(timestamp) {
		return new Date(timestamp).toUTCString().slice(-12,-7)
	});
	hbs.registerHelper("enumerate", function(context, options) {
	  var ret = "";

	  for (var i = 0, j = context.length; i < j; i++) {
	    // add separator with oxford comma
	    if (i === j-1 && i !== 0) ret += context.length>2 ? ", and " : " and "
	    else if (i !== 0) ret += ", "

	    ret = ret + options.fn(context[i]);
	  }

	  return ret;
	});
	hbs.registerHelper('getConfusionMatrixEntry', function(cm, realClassName, predictedClassName) {
		if (!cm[realClassName] || !cm[realClassName][predictedClassName])
			return 0
		return cm[realClassName][predictedClassName]
	});
}

