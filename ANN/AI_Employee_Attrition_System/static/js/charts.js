const ctx =
document.getElementById(
'attritionChart'
);

new Chart(ctx, {

type: 'pie',

data: {

labels: [
'Stayed',
'Resigned'
],

datasets: [{

data: [88,12]

}]

}

});