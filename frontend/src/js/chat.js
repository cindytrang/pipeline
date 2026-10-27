
const $messages = document.querySelectorAll(".message")
const $heatmap = document.querySelector(".heatmap .bar")
const $chart = document.querySelector(".chart");
const $canvas = document.querySelector("#chartCanvas");

function getHeatmapColor(prediction) {
    // Keep the red component at full intensity
    const red = 255;
    const green = 255 * (1 - prediction);
    const blue = 0;
    const alpha = 0.25 + (0.75 * prediction); // Optional: adjust for more vivid color at higher predictions
    return `rgba(${red}, ${green}, ${blue}, 0.25)`;
}

$heatmap.style = `background: linear-gradient(90deg, ${getHeatmapColor(0)}, ${getHeatmapColor(1)})`

// annotate risk level

const riskWindowSize =  parseInt(document.querySelector(".chat").dataset.riskWindowSize)
const riskThreshold =  parseInt(document.querySelector(".chat").dataset.riskThreshold)

const lastPredictions = new Array(riskWindowSize).fill(0)
let noWarningYet = true
$messages.forEach((msg, i) => {
	const prediction = parseFloat(msg.getAttribute("data-prediction"))
	if (prediction >= .5) {
		msg.classList.add("positive")
	}

	lastPredictions[i%lastPredictions.length] = prediction >= .5 ? 1 : 0
	const riskLevel = lastPredictions.reduce((a,b) => a+b)
	msg.setAttribute("data-risk-level", riskLevel)
	msg.setAttribute("data-risk-threshold", riskThreshold) // i.e. "skepticism"
	if (riskLevel >= riskThreshold && noWarningYet) {
		msg.classList.add("warning")
		noWarningYet = false
	}

	msg.style = `background: ${getHeatmapColor(prediction)};`
})


// prediction chart

const data = [...$messages].map(el =>
	parseFloat(el.getAttribute("data-prediction"))
)

const labels = data.map((prediction, i) => `Message ${i+1}`)
const dots = [
	pattern.draw('square', '#ff6384'),
	pattern.draw('circle', '#36a2eb'),
	pattern.draw('diamond', '#cc65fe'),
]


const ctx = $canvas.getContext('2d');



const config = {
	type: 'line',
	data: {
		labels,
		datasets: [{
			label: 'Prediction',
			steppedLine: false,
			data: data,
			borderColor: "transparent", // will be updated later
			fill: false,
		}]
	},
	backgroundColor: dots,
	options: {
		responsive: true,
		title: {
			display: true,
			text: "Predictions over time",
		},

	scales: {
		xAxes: [{
			// ticks: {
			// 	stepSize: 10,
			// },
			gridLines: {
				color: "rgba(0, 0, 0, 0)",
			}
		}],
		yAxes: [{
			ticks: {
				min: 0,
				max: 1,
			},
			gridLines: {
				color: "rgba(0, 0, 0, 0)",
			}
		}]
	},
	}
}

window.chart = new Chart(ctx, config);

function updateGradient() {
	var heatmapGradient = ctx.createLinearGradient(0,chart.height, 0,0);
	heatmapGradient.addColorStop(0, getHeatmapColor(0));
	heatmapGradient.addColorStop(1, getHeatmapColor(1));

	chart.data.datasets[0].borderColor = heatmapGradient
	chart.update();
}

window.addEventListener("resize", () => setTimeout(updateGradient, 0))
updateGradient()

$canvas.onclick = function(e) {
	var activePoints = chart.getElementsAtEvent(e);
	if (!activePoints.length) return;
	const index = activePoints[0]._index;+
	console.log("scrolling to message", index)
	const $msg = document.querySelectorAll('.message')[index];
	const elY = window.scrollY + $msg.getBoundingClientRect().top;
	window.scrollTo(window.scrollX, elY - 16);
};


const reqFrame = async () => new Promise(res => requestAnimationFrame(res));

const $line = document.querySelector(".chart .scrollIndicator, .chart .line");

const $segmentSeparators = document.querySelectorAll(".separator")
const $chat = document.querySelector(".chat");


function setMarker(y1, y2, className = "") {
	const chatHeight = $chat.offsetHeight
	const chatTop = $chat.offsetTop
	const progress1 = Math.min(Math.max(0, y1 - chatTop)/chatHeight, 1)
	const progress2 = Math.min(Math.max(0, y2 - chatTop)/chatHeight, 1)

	// const height = chart.height - 95;
	// const left = 33 + (progress1*(chart.width-65));
	// const width = Math.max(33 + (progress2*(chart.width-65)) - left, 1);

	const height = chart.height - 140;
	const left = 45 + (progress1*(chart.width-50));
	const width = Math.max(45 + (progress2*(chart.width-50)) - left, 1);
	const style = `
		left:   ${Math.round(left)}px;
		height: ${Math.round(height)}px;
		width:  ${Math.round(width)}px
	`

	const $marker = document.createElement("div")
	$marker.classList.add("line")
	$marker.classList.add(className)
	$marker.style = style

	$chart.appendChild($marker)
}

const draw = () => {
	document.querySelectorAll(".line").forEach(el => el.remove())

	if ($messages.length > 30) {
		setMarker(scrollY, scrollY+window.innerHeight, "scrollIndicator")
	}

	$segmentSeparators.forEach($seg => {
		const $msgBefore = $seg.nextElementSibling
		// const index = parseInt($msgBefore.id.replace(/\D/g, ""))
		setMarker($msgBefore.offsetTop, $msgBefore.offsetTop, "segmentMarker")
	})

}

const loop = async () => {
	while(true) {
		await reqFrame();
		draw();
	}
}

loop()
