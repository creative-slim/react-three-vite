import React, { useRef, useMemo, useCallback } from "react";
import { Canvas, useFrame, extend, useThree } from "@react-three/fiber";
import { Box, Environment, OrbitControls, Sphere, useTexture, shaderMaterial } from "@react-three/drei";
import { useControls } from "leva";
import { Color, RepeatWrapping, DoubleSide } from "three";

const waterVertexShader = `
  uniform float uTime;
  uniform float uNoiseFrequency;
  uniform float uNoiseAmplitude;
  uniform float uNoiseSpeed;

  varying vec2 vUv;
  varying vec3 vWorldPosition;
  varying vec3 vWorldNormal;

  // Simplex 3D noise
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.853734720909014 * r; }

  float snoise(vec3 v) {
    const vec2 C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 = v - i + dot(i, C.xxx) ;
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod289(i);
    vec4 p = permute( permute( permute(
              i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
            + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
            + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
    float n_ = 0.142857142857; // 1.0/7.0
    vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );
    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3) ) );
  }

  void main() {
    vUv = uv;

    // Use uniforms for noise properties
    float displacement = snoise(position * uNoiseFrequency + uTime * uNoiseSpeed) * uNoiseAmplitude;
    displacement += snoise(position * uNoiseFrequency * 2.5 + uTime * uNoiseSpeed * 1.5) * uNoiseAmplitude * 0.5;

    vec3 newPosition = position + normal * displacement;

    // Calculate world position and normal for reflection
    vec4 worldPosition = modelMatrix * vec4(newPosition, 1.0);
    vWorldPosition = worldPosition.xyz;
    vWorldNormal = normalize(mat3(modelMatrix) * normal); // Assuming no non-uniform scaling

    gl_Position = projectionMatrix * viewMatrix * worldPosition;
  }
`;

const waterFragmentShader = `
  uniform sampler2D uTexture;
  uniform bool uUseTexture; // To control texture usage
  uniform vec3 uColor;
  uniform float uOpacity;

  uniform samplerCube uEnvMap; 
  uniform float uReflectivity; 

  uniform float uRoughness;
  uniform float uMetalness;

  // Caustics Uniforms
  uniform float uTime;
  uniform float uCausticsFrequency;
  uniform float uCausticsSpeed;
  uniform float uCausticsIntensity;
  uniform float uCausticsSharpness;
  uniform float uCausticsEdgeThickness;
  uniform float uCausticsDistortionFrequency; // New
  uniform float uCausticsDistortionAmplitude; // New

  varying vec2 vUv;
  varying vec3 vWorldPosition;
  varying vec3 vWorldNormal;

  // Simplex 3D noise (copied from vertex shader)
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.853734720909014 * r; }

  float snoise(vec3 v) {
    const vec2 C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 = v - i + dot(i, C.xxx) ;
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod289(i);
    vec4 p = permute( permute( permute(
              i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
            + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
            + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
    float n_ = 0.142857142857; // 1.0/7.0
    vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );
    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3) ) );
  }

  // Hash function to create pseudo-random vectors
  vec2 hash( vec2 p ) {
      p = vec2( dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)) );
      return -1.0 + 2.0*fract(sin(p)*43758.5453123);
  }

  // Voronoi noise function returning distance to closest (F1) and second closest (F2) points
  vec2 voronoi(vec2 x, float time) {
      vec2 p = floor(x);
      vec2 f = fract(x);
      
      float F1 = 10.0; // Initialize with a large value
      float F2 = 10.0; // Initialize with a large value

      for (int j = -1; j <= 1; j++) {
          for (int i = -1; i <= 1; i++) {
              vec2 g = vec2(float(i), float(j)); // Neighbor cell
              vec2 o = hash(p + g); // Random offset for cell center
              
              float cellTime = time + dot(p + g, vec2(0.13, 0.27)) * 10.0; // Vary animation per cell
              vec2 animatedOffset = o + vec2(sin(cellTime * 0.5), cos(cellTime * 0.3)) * 0.4; // Adjust movement

              vec2 r = g + animatedOffset - f;
              float d = dot(r,r); // Squared distance

              if (d < F1) {
                  F2 = F1;
                  F1 = d;
              } else if (d < F2) {
                  F2 = d;
              }
          }
      }
      return vec2(sqrt(F1), sqrt(F2)); // Return actual distances
  }


  void main() {
    vec3 albedo = uColor;
    if (uUseTexture) {
        vec4 texSample = texture2D(uTexture, vUv);
        albedo = mix(uColor, texSample.rgb, texSample.a * 0.6 + 0.4); 
    }

    vec3 N = normalize(vWorldNormal);
    vec3 V = normalize(cameraPosition - vWorldPosition);
    vec3 R = reflect(-V, N);

    vec3 envColor = textureCube(uEnvMap, R).rgb;
    vec3 processedReflection = mix(envColor, albedo, uRoughness);

    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, uMetalness);

    vec3 diffuseColor = albedo * (1.0 - uMetalness); 
    vec3 specularColor = processedReflection * F0 * uReflectivity;
    
    vec3 finalColor = diffuseColor + specularColor;

    // Caustics UV Distortion
    float distortionTime = uTime * 0.1; // Slower animation for distortion
    vec2 distortionOffset = vec2(
      snoise(vec3(vUv * uCausticsDistortionFrequency, distortionTime)),
      snoise(vec3(vUv * uCausticsDistortionFrequency + vec2(5.2, 1.3), distortionTime)) // Offset for second noise
    ) * uCausticsDistortionAmplitude;
    vec2 distortedUv = vUv + distortionOffset;

    // Caustics Calculation using F2-F1 for edges
    vec2 voronoiDistances = voronoi(distortedUv * uCausticsFrequency, uTime * uCausticsSpeed);
    float f1 = voronoiDistances.x;
    float f2 = voronoiDistances.y;

    float edgeFactor = f2 - f1;
    
    // Gaussian-like profile for softer lines
    float normalizedEdge = edgeFactor / uCausticsEdgeThickness;
    float causticsPattern = exp(-normalizedEdge * normalizedEdge * uCausticsSharpness);
    
    float causticsEffect = causticsPattern * uCausticsIntensity;

    finalColor.rgb += causticsEffect; // Add caustics to the final color

    gl_FragColor = vec4(finalColor, uOpacity);
  }
`;

const WaterMaterial = shaderMaterial(
  {
    // Uniforms
    uTime: 0,
    uColor: new Color(0x1e90ff), // Default water color
    uTexture: null,
    uUseTexture: true, // Default to using texture if available
    uOpacity: 0.8, // Default opacity
    uNoiseFrequency: 2.0, // Default noise frequency
    uNoiseAmplitude: 0.05, // Default noise amplitude
    uNoiseSpeed: 0.3, // Default noise speed
    uEnvMap: null, // For reflections
    uReflectivity: 0.5, // Default reflectivity
    uRoughness: 0.5, // Default roughness
    uMetalness: 0.0, // Default metalness (0 for dielectric, 1 for metallic)
    // Caustics uniforms
    uCausticsFrequency: 10.0,
    uCausticsSpeed: 0.15,
    uCausticsIntensity: 0.4,
    uCausticsSharpness: 15.0, // Higher values = sharper falloff for Gaussian
    uCausticsEdgeThickness: 0.05, // Controls base width of Gaussian lines
    uCausticsDistortionFrequency: 5.0, // New
    uCausticsDistortionAmplitude: 0.03, // New
  },
  waterVertexShader,
  waterFragmentShader,
  material => {
    // Optional: Callback to configure the material
    if (material) {
      material.transparent = true;
      material.depthWrite = false;
      // material.side = DoubleSide;
    }
  }
);

extend({ WaterMaterial });

const Scene = () => {
  const materialRef = useRef();
  const waterTexture = useTexture("/water-texture.jpg");
  const { scene } = useThree(); // To access scene.environment

  const {
    noiseFrequency,
    noiseAmplitude,
    noiseSpeed,
    waterColor,
    waterOpacity,
    reflectivity,
    roughness,
    metalness,
    useTextureFlag, // Renamed to avoid conflict with hook name
    // Caustics controls
    causticsFrequency,
    causticsSpeed,
    causticsIntensity,
    causticsSharpness,
    causticsEdgeThickness,
    causticsDistortionFrequency, // New control
    causticsDistortionAmplitude, // New control
  } = useControls("Water Shader", {
    noiseFrequency: { value: 6.4, min: 0.1, max: 10, step: 0.1 },
    noiseAmplitude: { value: 0.02, min: 0.01, max: 0.5, step: 0.01 },
    noiseSpeed: { value: 0.5, min: 0.0, max: 2, step: 0.01 },
    waterColor: "#1e90ff",
    waterOpacity: { value: 0.9, min: 0, max: 1, step: 0.01 },
    reflectivity: { value: 0.3, min: 0, max: 1, step: 0.01 },
    roughness: { value: 0.5, min: 0, max: 1, step: 0.01 },
    metalness: { value: 0.0, min: 0, max: 1, step: 0.01 },
    useTextureFlag: { value: true, label: "Use Water Texture" },
    causticsFrequency: { value: 10.0, min: 1, max: 50, step: 0.1, folder: "Caustics" },
    causticsSpeed: { value: 0.15, min: 0.0, max: 1.0, step: 0.01, folder: "Caustics" },
    causticsIntensity: { value: 0.4, min: 0, max: 2, step: 0.01, folder: "Caustics" },
    causticsSharpness: { value: 15.0, min: 1, max: 100, step: 0.1, folder: "Caustics" }, // Increased max for Gaussian
    causticsEdgeThickness: { value: 0.05, min: 0.001, max: 0.5, step: 0.001, folder: "Caustics" },
    causticsDistortionFrequency: { value: 5.0, min: 0.1, max: 20, step: 0.1, folder: "Caustics" },
    causticsDistortionAmplitude: { value: 0.03, min: 0.0, max: 0.2, step: 0.001, folder: "Caustics" },
  });

  if (waterTexture) {
    waterTexture.wrapS = waterTexture.wrapT = RepeatWrapping;
  }

  useFrame((state, delta) => {
    if (materialRef.current) {
      materialRef.current.uniforms.uTime.value += delta; // uNoiseSpeed will now control the effective speed
    }
  });

  return (
    <>
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 15, 10]} intensity={1.5} castShadow />

      {/* Reduced segments for performance, adjust as needed. Original was 1300. */}
      <Sphere name='oceann' args={[1, 256, 256]} position={[0, 0, 0]}>
        <waterMaterial
          ref={materialRef}
          attach='material'
          uTexture={waterTexture} // Always pass the loaded texture
          uUseTexture={useTextureFlag && !!waterTexture} // Control usage in shader
          uColor={new Color(waterColor)}
          uOpacity={waterOpacity}
          uNoiseFrequency={noiseFrequency}
          uNoiseAmplitude={noiseAmplitude}
          uNoiseSpeed={noiseSpeed}
          uEnvMap={scene.environment} // Pass the environment map
          uReflectivity={reflectivity}
          uRoughness={roughness}
          uMetalness={metalness}
          // Pass caustics uniforms
          uCausticsFrequency={causticsFrequency}
          uCausticsSpeed={causticsSpeed}
          uCausticsIntensity={causticsIntensity}
          uCausticsSharpness={causticsSharpness}
          uCausticsEdgeThickness={causticsEdgeThickness}
          uCausticsDistortionFrequency={causticsDistortionFrequency} // Pass new uniform
          uCausticsDistortionAmplitude={causticsDistortionAmplitude} // Pass new uniform
        />
      </Sphere>
    </>
  );
};

const App = () => {
  const localModelUrl = "/artist_workshop_4k.hdr";

  return (
    <Canvas camera={{ fov: 70, position: [0, 0.5, 2.5] }}>
      {/* <Environment preset='sunset' background /> */}
      <Environment files={localModelUrl} background />
      <OrbitControls />
      <Scene />
    </Canvas>
  );
};

export default App;
