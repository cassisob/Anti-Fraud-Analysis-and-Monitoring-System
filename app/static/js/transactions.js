function formatTime(timestamp) {
    const date = new Date(timestamp);
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    return `${hours}:${minutes}`;
}

async function updateCharts() {

    var response = await fetch('/transactions/data', {
        method: 'GET'
    });

    const data = await response.json();

    const timestampsApprove = data.Approve.map(entry => entry[0]);
    const timestampsDenyModel = data.DenyModel.map(entry => entry[0]);
    const timestampsDenyRule = data.DenyRule.map(entry => entry[0]);

    updateChart('plotTipo1', data.Approve, 'Approve', timestampsApprove);
    updateChart('plotTipo2', data.DenyModel, 'Deny (Model)', timestampsDenyModel);
    updateChart('plotTipo3', data.DenyRule, 'Deny (Rule)', timestampsDenyRule);

    const approve_trend = data.Trend.Approve;
    const deny_model_trend = data.Trend.DenyModel;
    const deny_rule_trend = data.Trend.DenyRule;

    const approve_total = +data.Overall.Approve;
    const deny_model_total = +data.Overall.DenyModel;
    const deny_rule_total = +data.Overall.DenyRule;

    document.getElementById('qtdApprove').textContent = approve_total;
    document.getElementById('qtdDenyModel').textContent = deny_model_total;
    document.getElementById('qtdDenyRule').textContent = deny_rule_total;


    document.getElementById('totalQtd').textContent = approve_total + deny_model_total + deny_rule_total;

    updateTrends(approve_trend, 'iconApprove', 'trendApprove');
    updateTrends(deny_model_trend, 'iconDenyModel', 'trendDenyModel');
    updateTrends(deny_rule_trend, 'iconDenyRule', 'trendDenyRule');
}

function updateChart(container, data, name, timestamps) {

    const trace1 = {
        x: timestamps,
        y: data.map(entry => entry[1]),
        type: 'scatter',
        mode: 'lines+markers',
        name: name,
        line: { color: '#1f77b4' }
    };


    const layout = {
        title: {
            text: name
            , font: {
                family: 'Roboto, sans-serif'
                , size: 20
                , color: '#ffffff'
                , weight: 'bold'
                , margin: 0
            }
        }
        , paper_bgcolor: '#2b2b2b'
        , plot_bgcolor: '#2b2b2b'
        , font: {
            color: '#dcdcdc'
        }
        , xaxis: {
            title: 'Timeline'
            , tickmode: 'linear'
        }
        , yaxis: {
            title: 'Quantity'
            , range: [0, null] // <-- Always start Y axis at zero
        }
        , showlegend: false
    };

    Plotly.newPlot(container, [trace1], layout, { responsive: true, margin: 0 });
}

function updateTrends(trend_value, iconId, trendId) {
    const icon = document.getElementById(iconId);
    const trend = document.getElementById(trendId);

    if (trend_value == 2) {
        icon.className = 'fa-solid fa-arrow-up fa-beat fa-xl';
        trend.textContent = 'Rising a lot';
    } else if (trend_value == 1) {
        icon.className = 'fa-solid fa-arrow-trend-up fa-beat fa-xl';
        trend.textContent = 'Rising';
    } else if (trend_value == 0) {
        icon.className = 'fa-solid fa-arrow-right fa-beat fa-xl';
        trend.textContent = 'Stable';
    } else if (trend_value == -1) {
        icon.className = 'fa-solid fa-arrow-trend-down fa-beat fa-xl';
        trend.textContent = 'Falling';
    } else {
        icon.className = 'fa-solid fa-arrow-down fa-beat fa-xl';
        trend.textContent = 'Falling a lot';
    }
}

// Update every 10 seconds
setInterval(() => {
    updateCharts();
}, 10000);

updateCharts();