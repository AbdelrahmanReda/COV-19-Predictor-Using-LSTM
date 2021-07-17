
    $(document).ready(function () {

        $('.buttons-csv').addClass('btn-primary')


        $('#example2').DataTable(
            {
                dom: 'Bfrtip',
                buttons: [
                    'copy', 'csv', 'excel', 'pdf', 'print'
                ]
            }
        );

        $('#example').DataTable(
            {
                dom: 'Bfrtip',
                buttons: [
                    'copy', 'csv', 'excel', 'pdf', 'print'
                ]
            }
        );
    });


    $.fn.dataTable.Buttons.defaults.dom.button.className = 'btn btn-primary mtb-5';




    //functions to get forecast
    function getPrediction() {
        var xmlHttp = new XMLHttpRequest();
        xmlHttp.open("GET", 'http://127.0.0.1:8000/forecastConfirmedCases/', false); // false for synchronous request
        xmlHttp.send(null);
        return JSON.parse(xmlHttp.response);
    }

    //-------------------------


    //function to get actual data
    function getActualConfirmed() {
        var xmlHttp = new XMLHttpRequest();
        xmlHttp.open("GET", 'http://127.0.0.1:8000/getActualConfirmed/', false); // false for synchronous request
        xmlHttp.send(null);
        return JSON.parse(xmlHttp.response);
    }

    function getActualRecovered() {
        const xmlHttp = new XMLHttpRequest();
        xmlHttp.open("GET", 'http://127.0.0.1:8000/getActualRecovered/', false); // false for synchronous request
        xmlHttp.send(null);
        return JSON.parse(xmlHttp.response);
    }



    function getActualDeath() {
        const xmlHttp = new XMLHttpRequest();
        xmlHttp.open("GET", 'http://127.0.0.1:8000/getActualDeath/', false); // false for synchronous request
        xmlHttp.send(null);
        return JSON.parse(xmlHttp.response);
    }

    // ----------------------- //
    //functions to construct graphs

    function constructConfirmedGraph() {
        let predictedDate = getPrediction()
        const confirmedPrediction = [];
        const deathPrediction = [];
        for (let i = 0; i < predictedDate['date'].length; i++) {
            confirmedPrediction.push([new Date(predictedDate['date'][i]).getTime(), predictedDate['forecast-confirmed'][i]])
            deathPrediction.push([new Date(predictedDate['date'][i]).getTime(), 6.74423 + predictedDate['forecast-confirmed'][i] * 0.0451])
        }
        let confirmedActual = parseDate(getActualConfirmed(), 'date', 'confirmed')
        Highcharts.chart('container', configGraph(confirmedActual, '#ff8f00', confirmedPrediction, '#007f05'));
        populateForecastingTable(confirmedPrediction, '#confirmed-cases')
        constructDeathGraph(deathPrediction)
    }

    function constructDeathGraph(dethPrediction) {
        var deathActual = parseDate(getActualDeath(), 'date', 'death')
        Highcharts.chart('container2', configGraph(deathActual, '#db0000', dethPrediction, '#280000'));
        populateForecastingTable(dethPrediction, "#death-cases")
    }

    function parseDate(wholeData, dateAttr = "date", metric) {
        parsedDate = []
        for (let i = 0; i < wholeData[dateAttr].length; i++) {
            parsedDate.push([new Date(wholeData[dateAttr][i]).getTime(), wholeData[metric][i]])
        }
        return parsedDate
    }
    function configGraph(actualData, ActualDataSerierscolor, PredictionData, PredictionDataDataSerierscolor) {
        config = {
            tooltip: {
                xDateFormat: '%d/%m/%Y',
                shared: true,
                split: true,
                enabled: true
            },
            chart: {
                type: 'line',
                zoomType: 'x'
            },
            title: {
                text: 'Number of Egypt Death Confirmed Cases \n & Predicted Death'
            },
            subtitle: {
                text: 'Source: Johns Hopkins University'
            },
            yAxis: {
                title: {
                    text: 'Death Cases'
                }
            },
            xAxis: {
                type: 'datetime',
                plotLines: [{
                    color: '#181414',
                    width: 2,
                    value: PredictionData[0][0],
                    dashStyle: 'ShortDot'
                }]
            },
            legend: {
                layout: 'horizontal',
                align: 'center',
                verticalAlign: 'bottom',
                x: 0,
                y: 0
            },
            series: [{
                name: "Actual Death Cases",
                data: actualData,
                color: ActualDataSerierscolor,
            },
                {
                    name: "Forecast Death Cases",
                    data: PredictionData,
                    dashStyle: 'longdash',
                    color: PredictionDataDataSerierscolor
                }]
            , plotOptions: {
                series: {
                    compare: 'percent',
                    showInNavigator: true,
                    dataGrouping: {
                        enabled: false
                    },
                }
            },
        }
        return config
    }

    function populateForecastingTable(predictedDate, tableClassName) {
        for (let i = 0; i < predictedDate.length; i++) {
            tableRow = document.createElement('tr')
            CounterData = document.createElement('td')
            dateData = document.createElement('td')
            newConfirmedData = document.createElement('td')
            CounterData.append(i + 1)
            dateData.append(predictedDate[i][0])
            newConfirmedData.append(predictedDate[i][1])
            tableRow.append(CounterData)
            tableRow.append(dateData)
            tableRow.append(newConfirmedData)
            $(tableClassName).prepend(tableRow)
        }
    }


    // ----------------------- //


    constructConfirmedGraph()


