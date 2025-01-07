// Initialize the map centered on a general region to include all locations
const map = L.map('map').setView([15.9129, 79.7400], 6); // Centered to fit all schools

// Add OpenStreetMap tiles
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 18,
  attribution: '© OpenStreetMap contributors'
}).addTo(map);

// Define a custom icon resembling the Google Maps pin
const googlePinIcon = L.icon({
  iconUrl: 'https://maps.google.com/mapfiles/kml/shapes/placemark_circle.png',
  iconSize: [32, 32],
  iconAnchor: [16, 32],
  popupAnchor: [0, -32]
});

// Add an HTML container for displaying the school box
const schoolBox = document.createElement('div');
schoolBox.id = 'school-box';
schoolBox.style.position = 'absolute';
schoolBox.style.display = 'none'; // Hidden by default
schoolBox.style.zIndex = 1000;
schoolBox.style.width = '250px';
schoolBox.style.background = 'white';
schoolBox.style.border = '1px solid #ccc';
schoolBox.style.borderRadius = '8px';
schoolBox.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.2)';
schoolBox.style.padding = '10px';
schoolBox.style.textAlign = 'center';
document.body.appendChild(schoolBox);

// Function to show the school box with dynamic content
function showSchoolBox(e, schoolName, imagePath) {
  const coords = e.containerPoint;
  schoolBox.style.left = `${coords.x + 20}px`; // Position slightly offset from the cursor
  schoolBox.style.top = `${coords.y - 150}px`;
  schoolBox.style.display = 'block';
  schoolBox.innerHTML = `
    <div style="font-weight: bold; font-size: 18px; margin-bottom: 10px;">${schoolName}</div>
    <img src="${imagePath}" alt="${schoolName}" style="width:100%;height:auto;border-radius:5px;">
  `;
}

// Function to hide the school box on mouseout
function hideSchoolBox() {
  schoolBox.style.display = 'none';
}

// Add markers for all schools
const schools = [
  { name: 'Oakridge International School', coords: [12.8874546, 77.7525313], image: 'images/oakridge.jpg',},
  { name: 'The CHIREC International School', coords: [13.076539863579558, 77.73896274818894], image: './images/chirec.jpg' },
  { name: 'The Aga Khan Academy', coords: [17.246731921803573, 78.48058103079558], image: './images/aga_khan.jpg' },
  { name: 'Indus International School', coords: [12.835299540563238, 77.77850776926492], image: './images/indus.jpg' },
  { name: 'GD Goenka School', coords: [17.36170946076783, 78.56383857308079], image: './images/goenka.jpg' },
  { name: 'NIHOC The International School', coords: [17.396194814998562, 78.62564163110986], image: './images/nihoc.jpg' },
  { name: 'Sancta Maria School', coords: [17.607169462355436, 78.40693523875808], image: './images/sancta_maria.jpg' },
  { name: 'International School of Hyderabad', coords: [17.542697883238592, 78.28683287949816], image: './images/ish.jpg' },
  { name: 'Manthan International School', coords: [17.460292395334818, 78.29846158995463], image: './images/manthan.jpg' },
  { name: 'Rishi Valley School', coords: [13.63638709376012, 78.45389756450555], image: './images/rishi_valley.jpg' },
  { name: 'Ambitus World School', coords: [16.542942593949597, 80.6723706311272], image: './images/ambitus.jpg' },
  { name: 'The Vizag International School', coords: [18.064579966081098, 83.33420476163535], image: './images/vizag.jpg' },
  { name: 'Candor International School', coords: [12.82021005514595, 77.62189727618028], image: './images/candor.jpg' },
  { name: 'The Green School', coords: [12.94563760981085, 77.78634957552616], image: './images/green.jpg' },
  { name: 'Inventure Academy', coords: [12.895413764458958, 77.74419758962246], image: './images/inventure.jpg' },
  { name: 'Pathways World School', coords: [28.312711895906258, 77.01278390505134], image: './images/pathways.jpg' },
  { name: 'Ecole Mondiale World School', coords: [19.113143387849235, 72.83429774545095], image: './images/ecole_mondiale.jpg' },
  { name: 'Tridha', coords: [19.122906737081873, 72.85904558978828], image: './images/tridha.jpg' },
  { name: 'Woodstock School', coords: [30.453965220447383, 78.10085140517408], image: './images/woodstock.jpg' },
  { name: 'Alpha International School', coords: [12.924296953124166, 80.15830999869962], image: './images/alpha.jpg' },
  { name: 'Shiv Nadar School', coords: [13.005156533030055, 80.26326775927699], image: './images/shiv_nadar.jpg' },
  { name: 'Ørestad High School', coords: [55.63183195977894, 12.58110280982054], image: './images/orestad.jpg' }
];

// Loop through the schools and create markers
schools.forEach(school => {
  const marker = L.marker(school.coords, { icon: googlePinIcon }).addTo(map);
  marker.on('mouseover', function (e) {
    showSchoolBox(e, school.name, school.image);
  });
  marker.on('mouseout', hideSchoolBox);
});