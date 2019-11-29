/**
 * 
 * Credits: https://dev.to/maniflames/creating-a-custom-shader-in-threejs-3bhi
 */

window.addEventListener('load', init);
window.addEventListener( 'mousemove', onMouseMove, false );
let scene;
let camera;
let renderer;

var raycaster = new THREE.Raycaster();
var mouse = new THREE.Vector2();
var light = new THREE.PointLight( 0xff0000, 1, 100 );
light.position.set( 50, 50, 50 );
light.power = 200;
light.shadow = 200;


let uniforms = {
    colorA: { type: 'vec3', value: new THREE.Color(0xff0000) },
    colorB: { type: 'vec3', value: new THREE.Color(0x00ff00) },
    time: { type: 'float', value: 0 },
    mouse: {type: 'vec2', value: new THREE.Vector2()}
};

let vertexShader = `
    varying vec4 modelViewPosition; 
    uniform float time; 

    void main() {
        modelViewPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * modelViewPosition; 
    }
`;

let fragmentShader = `
    uniform vec3 colorA; 
    uniform vec3 colorB; 
    uniform float time;
    uniform vec2 mouse;

    void main() {
        vec3 finalColor = colorA + colorB;
        finalColor.x = mouse.y;
        finalColor.y = mouse.x;
        gl_FragColor = vec4(finalColor, 1.0);
    }
`;

function init() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 5;

    renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);
    
    addBasicCube();
    addShadingCube();

    animationLoop();
}

function addBasicCube() {
    let geometry = new THREE.BoxGeometry(1, 1, 1);
    var material = new THREE.MeshBasicMaterial({color: 0x0000ff});

    let mesh = new THREE.Mesh(geometry, material);
    mesh.position.x = -2;
    scene.add(mesh);
    scene.add( light );
}

function addShadingCube() {
    let geometry = new THREE.BoxGeometry(1, 1, 1);

    let material = new THREE.ShaderMaterial({
        uniforms: uniforms,
        fragmentShader: fragmentShader,
        vertexShader: vertexShader,
    });

    let mesh = new THREE.Mesh(geometry, material);

    mesh.position.x = 2;
    scene.add(mesh);
}

function animationLoop() {

	// update the picking ray with the camera and mouse position
	raycaster.setFromCamera( mouse, camera );

	// calculate objects intersecting the picking ray
	var intersects = raycaster.intersectObjects( scene.children );

	for ( var i = 0; i < intersects.length; i++ ) {

	    intersects[ i ].object.rotation.x += 0.2;
	}

    renderer.render(scene, camera);

    uniforms.time.value += 0.1;

    scene.traverse(function(node) {
        if(node instanceof THREE.Mesh) {
            node.rotation.x += 0.01;
            node.rotation.y += 0.03;
        }
    });

    requestAnimationFrame(animationLoop);
}

function onMouseMove( event ) {

	// calculate mouse position in normalized device coordinates
	// (-1 to +1) for both components

	mouse.x = ( event.clientX / window.innerWidth ) * 2 - 1;
	mouse.y = - ( event.clientY / window.innerHeight ) * 2 + 1;
    uniforms.mouse.value.x = mouse.x;
    uniforms.mouse.value.y = mouse.y;
    light.position.set( mouse.x, mouse.y, mouse.x )
}
