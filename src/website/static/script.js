// src/website/static/script.js

let availableModels = {};
let availableDatasets = [];

document.addEventListener('DOMContentLoaded', async () => {
    await fetchAvailableModels();
    await fetchAvailableDatasets();
    updateUIForSelectedModel();
});

async function fetchAvailableModels() {
    try {
        const response = await fetch('/models');
        availableModels = await response.json();
        const selector = document.getElementById('model-selector');
        selector.innerHTML = '';
        for (const key in availableModels) {
            const option = document.createElement('option');
            option.value = key;
            option.textContent = `${key.toUpperCase()}: ${availableModels[key].description}`;
            selector.appendChild(option);
        }
    } catch (error) {
        console.error("Error fetching models:", error);
    }
}

async function fetchAvailableDatasets() {
    try {
        const response = await fetch('/datasets');
        availableDatasets = await response.json();
        const selector = document.getElementById('dataset-selector');
        selector.innerHTML = '';
        availableDatasets.forEach(ds => {
            const option = document.createElement('option');
            option.value = ds.name;
            option.textContent = `${ds.name} (${ds.cases_count} cases)`;
            selector.appendChild(option);
        });
        
        // Update max for case index if datasets exist
        if (availableDatasets.length > 0) {
            document.getElementById('case-index').max = availableDatasets[0].cases_count - 1;
        }
        
        selector.onchange = (e) => {
            const ds = availableDatasets.find(d => d.name === e.target.value);
            if (ds) {
                document.getElementById('case-index').max = ds.cases_count - 1;
            }
        };
    } catch (error) {
        console.error("Error fetching datasets:", error);
    }
}

function updateUIForSelectedModel() {
    const selectedKey = document.getElementById('model-selector').value;
    if (!selectedKey || !availableModels[selectedKey]) return;

    // Clear old plots when switching models
    document.getElementById('plot-container').innerHTML = '';
    const inputPlotContainer = document.getElementById('input-plot-container');
    if (inputPlotContainer) inputPlotContainer.innerHTML = '';

    const config = availableModels[selectedKey];

    // Show/hide scalar section
    const scalarWrapper = document.getElementById('scalar-inputs-wrapper');
    if (config.input_scalars.length > 0) {
        scalarWrapper.classList.remove('hidden');
        createScalarInputs(config.input_scalars);
    } else {
        scalarWrapper.classList.add('hidden');
    }

    // Show/hide field section
    const fieldWrapper = document.getElementById('field-inputs-wrapper');
    if (config.input_fields.length > 0) {
        fieldWrapper.classList.remove('hidden');
        document.getElementById('field-label').textContent = `Field Data (${config.input_fields.join(', ') || 'None'}):`;
        document.getElementById('field_data_flat').placeholder = `Requires ${config.input_fields.length} field(s). Please load a dataset case above.`;
    } else {
        fieldWrapper.classList.add('hidden');
    }
}

function createScalarInputs(scalars) {
    const container = document.getElementById('scalar-inputs-container');
    container.innerHTML = ''; // Clear old sliders

    scalars.forEach(scalar => {
        const row = document.createElement('div');
        row.className = 'scalar-input-row';
        row.innerHTML = `
            <label for="slider-${scalar.name}">${scalar.name} [${scalar.min} - ${scalar.max}]:</label>
            <input type="range" id="slider-${scalar.name}" name="${scalar.name}" min="${scalar.min}" max="${scalar.max}" step="${scalar.step}" value="${scalar.default}">
            <input type="number" id="number-${scalar.name}" min="${scalar.min}" max="${scalar.max}" step="${scalar.step}" value="${scalar.default}">
        `;
        container.appendChild(row);

        const slider = row.querySelector(`#slider-${scalar.name}`);
        const numberInput = row.querySelector(`#number-${scalar.name}`);

        const triggerUpdate = () => {
            if (window.debounceTimer) clearTimeout(window.debounceTimer);
            window.debounceTimer = setTimeout(() => sendPredictionRequest(), 300);
        };

        slider.oninput = () => { numberInput.value = slider.value; triggerUpdate(); };
        numberInput.oninput = () => { slider.value = numberInput.value; triggerUpdate(); };
    });
}



async function loadCaseData() {
    const datasetName = document.getElementById('dataset-selector').value;
    const caseIndex = document.getElementById('case-index').value;
    const selectedKey = document.getElementById('model-selector').value;
    const config = availableModels[selectedKey];
    
    if (!datasetName) return;
    
    try {
        const response = await fetch(`/dataset/${datasetName}/case/${caseIndex}`);
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.detail || 'API request failed.');
        
        // Populate scalars
        config.input_scalars.forEach(scalar => {
            const slider = document.getElementById(`slider-${scalar.name}`);
            const numInput = document.getElementById(`number-${scalar.name}`);
            if (slider && numInput && data.scalars[scalar.name] !== undefined) {
                // Ensure value is within bounds or update bounds if necessary
                const val = data.scalars[scalar.name];
                slider.value = val;
                numInput.value = val;
            }
        });
        
        // Populate fields
        let allFields = [];
        config.input_fields.forEach(fieldName => {
            if (data.fields[fieldName]) {
                allFields = allFields.concat(data.fields[fieldName]);
            } else {
                console.warn(`Field ${fieldName} not found in dataset ${datasetName}`);
                // fallback to zeros if missing
                allFields = allFields.concat(new Array(11 * 401).fill(0));
            }
        });
        
        if (config.input_fields.length > 0) {
            document.getElementById('field_data_flat').value = allFields.join(',');
        }
        
        plotInputs();
    } catch (error) {
        alert(`Error loading case: ${error.message}`);
        console.error('Failed to load case data:', error);
    }
}

async function sendPredictionRequest() {
    const selectedKey = document.getElementById('model-selector').value;
    const config = availableModels[selectedKey];

    // We REMOVED the reference to jsonOutput here
    const plotContainer = document.getElementById('plot-container');
    const loadingIndicator = document.getElementById('loading');

    // We REMOVED jsonOutput.textContent = '' here
    // No longer clearing innerHTML to allow Plotly.react to update in-place
    const fieldInput = document.getElementById('field_data_flat').value;
    if (config.input_fields.length > 0 && !fieldInput) {
        alert("Please load a case from the dataset first to provide the required field data (e.g., channel topography).");
        return;
    }
    
    loadingIndicator.classList.remove('hidden');
    
    plotInputs();

    try {
        // Gather scalar inputs dynamically
        const scalar_features = config.input_scalars.map(scalar => {
            return parseFloat(document.getElementById(`slider-${scalar.name}`).value);
        });

        // Gather field inputs
        const fieldInput = document.getElementById('field_data_flat').value;
        const field_data_flat = config.input_fields.length > 0 ? fieldInput.split(',').map(Number) : [];

        const response = await fetch(`/predict/${selectedKey}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scalar_features, field_data_flat }),
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'API request failed.');

        // --- THIS BLOCK IS THE MAIN CHANGE ---
        // Log the summary to the developer console for debugging, not the webpage
        const summaryData = { ...data };
        delete summaryData.field_predictions_flat;
        console.log("API Response Summary:", summaryData);

        plotResults(data, config);

    } catch (error) {
        // --- THIS IS THE OTHER CHANGE ---
        // Use an alert to show errors, since our display box is gone
        alert(`Error: ${error.message}`);
        console.error('Prediction failed:', error);
    } finally {
        loadingIndicator.classList.add('hidden');
    }
}

function plotResults(data, config) {
    const plotContainer = document.getElementById('plot-container');

    if (data.scalar_predictions && data.scalar_predictions.length > 0) {
        // Create a container for scalars if it doesn't exist
        let scalarDiv = document.getElementById('plot-scalars');
        if (!scalarDiv) {
            scalarDiv = document.createElement('div');
            scalarDiv.id = 'plot-scalars';
            plotContainer.appendChild(scalarDiv);
        }
        Plotly.newPlot(scalarDiv.id, [{
            x: config.output_scalars,
            y: data.scalar_predictions,
            type: 'bar'
        }], { title: 'Predicted Scalar Values' });
    }

    if (data.field_predictions_flat) {
        const gridHeight = 11;
        const gridWidth = 401;

        for (const fieldName in data.field_predictions_flat) {
            const flatZ = data.field_predictions_flat[fieldName];
            const minZ = Math.min(...flatZ).toFixed(4);
            const maxZ = Math.max(...flatZ).toFixed(4);
            const zData = unflatten(flatZ, gridHeight, gridWidth);

            const divId = `plot-result-${fieldName}`;
            let plotDiv = document.getElementById(divId);
            if (!plotDiv) {
                plotDiv = document.createElement('div');
                plotDiv.id = divId;
                plotContainer.appendChild(plotDiv);
            }

            // --- THIS IS THE FIX FOR ASPECT RATIO ---
            const layout = {
                title: `Predicted ${fieldName} (Min: ${minZ}, Max: ${maxZ})`,
                width: plotContainer.clientWidth, // Use the container's width
                height: 250, // A fixed, shorter height
                xaxis: { title: 'Channel Length (Grid Points)' },
                yaxis: {
                    title: 'Width',
                    autorange: 'reversed',
                    // This forces the y-axis scaling to respect the x-axis
                    scaleanchor: 'x',
                    scaleratio: 75 / (gridWidth / gridHeight) // Adjust this ratio to tune the appearance
                }
            };

            Plotly.react(plotDiv, [{ z: zData, type: 'heatmap', colorscale: 'Viridis' }], layout);
        }
    }
}

function unflatten(arr, rows, cols) {
    const newArr = [];
    for (let r = 0; r < rows; r++) {
        newArr.push(arr.slice(r * cols, (r + 1) * cols));
    }
    return newArr;
}

function plotInputs() {
    const selectedKey = document.getElementById('model-selector').value;
    const config = availableModels[selectedKey];
    const fieldInput = document.getElementById('field_data_flat').value;
    const plotContainer = document.getElementById('input-plot-container');
    
    if (!plotContainer) return;
    
    if (!fieldInput || config.input_fields.length === 0) {
        plotContainer.innerHTML = '';
        return;
    }
    
    const fieldDataFlat = fieldInput.split(',').map(Number);
    const gridHeight = 11;
    const gridWidth = 401;
    const pointsPerField = gridHeight * gridWidth;
    
    // Only clear if we are generating different number of fields, but it's fine to just let react handle it.
    // However, if the model changes, we should clear it. We'll leave it as is for now, but remove the innerHTML clear to avoid flashing.
    // plotContainer.innerHTML = ''; // REMOVED TO PREVENT FLASHING
    
    for (let i = 0; i < config.input_fields.length; i++) {
        const fieldName = config.input_fields[i];
        const fieldSlice = fieldDataFlat.slice(i * pointsPerField, (i + 1) * pointsPerField);
        const zData = unflatten(fieldSlice, gridHeight, gridWidth);
        
        const divId = `plot-input-${fieldName}`;
        let plotDiv = document.getElementById(divId);
        
        if (!plotDiv) {
            plotDiv = document.createElement('div');
            plotDiv.id = divId;
            plotContainer.appendChild(plotDiv);
        }
        
        const layout = {
            title: `Input Field: ${fieldName}`,
            width: plotContainer.clientWidth,
            height: 250,
            xaxis: { title: 'Channel Length (Grid Points)' },
            yaxis: {
                title: 'Width',
                autorange: 'reversed',
                scaleanchor: 'x',
                scaleratio: 75 / (gridWidth / gridHeight)
            }
        };
        Plotly.react(plotDiv, [{ z: zData, type: 'heatmap', colorscale: 'Cividis' }], layout);
    }
}
