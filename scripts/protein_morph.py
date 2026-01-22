import os
import argparse
import numpy as np
import imageio_ffmpeg

from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from Bio.PDB import PDBParser, PDBIO, Superimposer, Structure, Model, Chain

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()


class ProteinMorph:
    def __init__(self, pdb_files):
        self.pdb_files = pdb_files
        self.parser = PDBParser(QUIET=True)
        self.structures = []
        self.aligned_structures = []

    def load_structures(self):
        for pdb_file in self.pdb_files:
            structure = self.parser.get_structure('protein', pdb_file)
            self.structures.append(structure)
        print(f"Loaded {len(self.structures)} structures")

    def align_structures(self):
        if not self.structures:
            raise ValueError("No structures loaded to align.")

        reference = self.structures[0]
        ref_atoms = [atom for atom in reference.get_atoms() if atom.get_name() == 'CA']

        self.aligned_structures = [reference]

        sup = Superimposer()

        for i, structure in enumerate(self.structures[1:], start=1):
            mobile_atoms = [atom for atom in structure.get_atoms() if atom.get_name() == 'CA']
            
            if len(ref_atoms) != len(mobile_atoms):
                print(f"Warning: Structure {i} has different number of CA atoms")
                min_len = min(len(ref_atoms), len(mobile_atoms))
                sup.set_atoms(ref_atoms[:min_len], mobile_atoms[:min_len])
            else:
                sup.set_atoms(ref_atoms, mobile_atoms)
            
            sup.apply(structure.get_atoms())
            self.aligned_structures.append(structure)
            print(f"Aligned structure {i}, RMSD: {sup.rms:.3f} Å")

    def interpolate_between_structures(self, struct1, struct2, num_frames=10):
        interpolated = []

        for frame in range(num_frames):
            alpha = frame / (num_frames - 1) if num_frames > 1 else 0
            
            # Create new structure
            new_structure = Structure.Structure(f'frame_{frame}')
            new_model = Model.Model(0)
            new_structure.add(new_model)

            for chain1 in struct1[0]:
                chain2 = struct2[0][chain1.id]
                new_chain = Chain.Chain(chain1.id)
                new_model.add(new_chain)

                for residue1 in chain1:
                    res_id = residue1.id
                    if res_id not in chain2:
                        continue
                    residue2 = chain2[res_id]
                    
                    new_residue = residue1.copy()
                    new_residue.detach_parent()
                    new_chain.add(new_residue)
                    
                    # Interpolate atom coordinates
                    for atom1 in residue1:
                        atom_name = atom1.get_name()
                        if atom_name in residue2:
                            atom2 = residue2[atom_name]
                            new_coord = (1 - alpha) * atom1.coord + alpha * atom2.coord
                            new_residue[atom_name].set_coord(new_coord)

            interpolated.append(new_structure)

        return interpolated

    def generate_morph(self, frames_between=10, output_dir='morph_frames'):
        if not self.aligned_structures:
            print("Aligning structures first...")
            self.align_structures()
        
        Path(output_dir).mkdir(exist_ok=True)
        
        all_frames = []
        self.frame_source_names = []
        frame_count = 0

        # Generate interpolated frames between each pair of aligned structures
        for i in range(len(self.aligned_structures) - 1):
            struct1 = self.aligned_structures[i]
            struct2 = self.aligned_structures[i + 1]

            name1 = Path(self.pdb_files[i]).stem
            name2 = Path(self.pdb_files[i+1]).stem

            interpolated = self.interpolate_between_structures(
                struct1, struct2, frames_between
            )
            
            # Save frames (skip last frame to avoid duplicates except for final structure)
            frames_to_save = interpolated if i == len(self.aligned_structures) - 2 else interpolated[:-1]

            for frame in frames_to_save:
                output_file = os.path.join(output_dir, f'frame_{frame_count:04d}.pdb')
                io = PDBIO()
                io.set_structure(frame)
                io.save(output_file)

                self.frame_source_names.append(f"{name1} -> {name2}")

                frame_count += 1
                all_frames.append(frame)

        print(f"\nGenerated {frame_count} frames in '{output_dir}'")
        return all_frames
        
    def create_pymol_script(self, output_dir='morph_output', script_name='view_morph.pml'):
        """Create a PyMOL script to visualize the morph."""
        pdb_files = sorted(Path(output_dir).glob('frame_*.pdb'))
        
        with open(script_name, 'w') as f:
            f.write("# CAAT Morph Script\n")
            f.write("reinitialize\n")
            f.write("bg_color white\n")
            f.write("set antialias, 1\n\n")
            
            for i, pdb_file in enumerate(pdb_files):
                f.write(f"load {pdb_file.as_posix()}, morph, {i+1}\n")
            
            f.write("\n# Style settings\n")
            f.write("hide everything\n")
            f.write("show cartoon, morph\n")
            f.write("spectrum count, rainbow, morph, byres=1\n\n")
            
            f.write("# Fixed screen label\n")
            f.write("pseudoatom screen_label\n")
            f.write("set label_screen_pts, 1\n")
            f.write("set label_position, [-1.5, 1.5, 0]\n")
            f.write("set label_size, 30\n")
            f.write("set label_color, black\n")
            f.write("set label_font_id, 7\n")
            
            for i, pdb_file in enumerate(pdb_files):
                clean_name = pdb_file.stem
                f.write(f'mdo {i+1}: label screen_label, "{clean_name}"\n')
            
            f.write("\n# Export settings\n")
            f.write("set movie_fps, 30\n")
            f.write("rewind\n")
            f.write("mplay\n")
            
        print(f"✅ Created PyMOL script: {script_name}")

    def create_animation(self, output_dir='morph_frames', output_file='morph_animation.mp4', 
                        format='mp4', fps=30, dpi=100, view_angle=(30, 45)):
        """
        Create MP4 or GIF animation from generated frames.
        
        Args:
            output_dir: Directory containing frame PDB files
            output_file: Output filename (e.g., 'animation.mp4' or 'animation.gif')
            format: 'mp4' or 'gif'
            fps: Frames per second
            dpi: Resolution (higher = better quality but larger file)
            view_angle: Tuple of (elevation, azimuth) for 3D view
        """
        pdb_files = sorted(Path(output_dir).glob('frame_*.pdb'))
        
        if not pdb_files:
            print(f"Error: No frame files found in '{output_dir}'")
            return
        
        print(f"Loading {len(pdb_files)} frames for animation...")
        
        # Load all frames
        frames_data = []
        for pdb_file in pdb_files:
            structure = self.parser.get_structure('frame', pdb_file)
            ca_coords = np.array([atom.coord for atom in structure.get_atoms() 
                                 if atom.get_name() == 'CA'])
            frames_data.append(ca_coords)
        
        # Set up the figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get coordinate bounds for consistent scaling
        all_coords = np.vstack(frames_data)
        x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
        y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
        z_min, z_max = all_coords[:, 2].min(), all_coords[:, 2].max()
        
        # Add some padding
        padding = 5
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_zlim(z_min - padding, z_max + padding)
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title('Protein Morph Animation')
        
        # Set initial view angle
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # Initialize plot objects
        line, = ax.plot([], [], [], 'o-', markersize=3, linewidth=1, color='blue', alpha=0.6)
        frame_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
        
        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            frame_text.set_text('')
            return line, frame_text
        
        def update(frame_idx):
            coords = frames_data[frame_idx]
            line.set_data(coords[:, 0], coords[:, 1])
            line.set_3d_properties(coords[:, 2])
            source_name = self.frame_source_names[frame_idx] if hasattr(self, 'frame_source_names') else ""
            frame_text.set_text(f'{source_name}\nFrame: {frame_idx + 1}/{len(frames_data)}')
            return line, frame_text
        
        # Create animation
        anim = FuncAnimation(fig, update, init_func=init, frames=len(frames_data),
                           interval=1000/fps, blit=True, repeat=True)
        
        # Save animation
        print(f"Generating {format.upper()} animation...")
        
        if format.lower() == 'gif':
            writer = PillowWriter(fps=fps)
            anim.save(output_file, writer=writer, dpi=dpi)
        elif format.lower() == 'mp4':
            try:
                writer = FFMpegWriter(fps=fps, bitrate=1800)
                anim.save(output_file, writer=writer, dpi=dpi)
            except Exception as e:
                print(f"Error: FFMpeg not found. {e}")
                print("Falling back to GIF format...")
                output_file = output_file.replace('.mp4', '.gif')
                writer = PillowWriter(fps=fps)
                anim.save(output_file, writer=writer, dpi=dpi)
        
        plt.close()
        print(f"Animation saved to: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    def create_rotating_animation(self, output_dir='morph_frames', output_file='morph_rotating.mp4',
                                 format='mp4', fps=30, dpi=100, rotation_speed=2):
        """
        Create animation with rotating view.
        
        Args:
            output_dir: Directory containing frame PDB files
            output_file: Output filename
            format: 'mp4' or 'gif'
            fps: Frames per second
            dpi: Resolution
            rotation_speed: Degrees per frame to rotate
        """
        pdb_files = sorted(Path(output_dir).glob('frame_*.pdb'))
        
        if not pdb_files:
            print(f"Error: No frame files found in '{output_dir}'")
            return
        
        print(f"Loading {len(pdb_files)} frames for rotating animation...")
        
        # Load all frames
        frames_data = []
        for pdb_file in pdb_files:
            structure = self.parser.get_structure('frame', pdb_file)
            ca_coords = np.array([atom.coord for atom in structure.get_atoms() 
                                 if atom.get_name() == 'CA'])
            frames_data.append(ca_coords)
        
        # Set up the figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get coordinate bounds
        all_coords = np.vstack(frames_data)
        x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
        y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
        z_min, z_max = all_coords[:, 2].min(), all_coords[:, 2].max()
        
        padding = 5
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_zlim(z_min - padding, z_max + padding)
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title('Protein Morph Animation (Rotating View)')
        
        line, = ax.plot([], [], [], 'o-', markersize=3, linewidth=1, color='blue', alpha=0.6)
        frame_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
        
        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            frame_text.set_text('')
            return line, frame_text
        
        def update(frame_idx):
            # Update protein structure
            coords = frames_data[frame_idx % len(frames_data)]
            line.set_data(coords[:, 0], coords[:, 1])
            line.set_3d_properties(coords[:, 2])
            
            # Rotate view
            azim = (frame_idx * rotation_speed) % 360
            ax.view_init(elev=30, azim=azim)
            
            frame_text.set_text(f'Frame: {(frame_idx % len(frames_data)) + 1}/{len(frames_data)}')
            return line, frame_text
        
        # Make animation longer to show full rotation
        total_frames = max(len(frames_data), int(360 / rotation_speed))
        
        anim = FuncAnimation(fig, update, init_func=init, frames=total_frames,
                           interval=1000/fps, blit=True, repeat=True)
        
        print(f"Generating {format.upper()} rotating animation...")
        
        if format.lower() == 'gif':
            writer = PillowWriter(fps=fps)
            anim.save(output_file, writer=writer, dpi=dpi)
        elif format.lower() == 'mp4':
            try:
                writer = FFMpegWriter(fps=fps, bitrate=1800)
                anim.save(output_file, writer=writer, dpi=dpi)
            except Exception as e:
                print(f"Error: FFMpeg not found. {e}")
                print("Falling back to GIF format...")
                output_file = output_file.replace('.mp4', '.gif')
                writer = PillowWriter(fps=fps)
                anim.save(output_file, writer=writer, dpi=dpi)
        
        plt.close()
        print(f"Rotating animation saved to: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Visualize protein structure morphology through interpolation.")
    argparser.add_argument(
        "--structures",
        type=str,
        required=True,
        help="Folder containing PDB files."
    )
    argparser.add_argument(
        "--frames",
        type=int,
        default=5,
        help="Number of interpolated frames between each structure pair (default: 5)."
    )
    argparser.add_argument(
        "--output",
        type=str,
        default="morph_frames",
        help="Output directory for generated frames (default: morph_frames)."
    )
    argparser.add_argument(
        "--pymol-script",
        type=str,
        default="view_morph.pml",
        help="Name of PyMOL script to generate (default: view_morph.pml)."
    )
    argparser.add_argument(
        "--create-video",
        action="store_true",
        help="Create MP4 video animation."
    )
    argparser.add_argument(
        "--create-gif",
        action="store_true",
        help="Create GIF animation."
    )
    argparser.add_argument(
        "--rotating",
        action="store_true",
        help="Create rotating view animation."
    )
    argparser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for video/gif (default: 30)."
    )
    argparser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Resolution/DPI for video/gif (default: 100)."
    )
    
    args = argparser.parse_args()
    
    # Find all PDB files in the specified folder
    structure_folder = Path(args.structures)
    if not structure_folder.exists():
        print(f"Error: Folder '{args.structures}' does not exist.")
        exit(1)
    
    pdb_files = sorted(structure_folder.glob('*.pdb'))
    
    if not pdb_files:
        print(f"Error: No PDB files found in '{args.structures}'")
        exit(1)
    
    print(f"Found {len(pdb_files)} PDB files in '{args.structures}'")
    
    # Create morph
    morph = ProteinMorph(pdb_files)
    morph.load_structures()
    morph.align_structures()
    morph.generate_morph(frames_between=args.frames, output_dir=args.output)
    morph.create_pymol_script(output_dir=args.output, script_name=args.pymol_script)

    if args.create_video:
        if args.rotating:
            morph.create_rotating_animation(
                output_dir=args.output,
                output_file='morph_rotating.mp4',
                format='mp4',
                fps=args.fps,
                dpi=args.dpi
            )
        else:
            morph.create_animation(
                output_dir=args.output,
                output_file='morph_animation.mp4',
                format='mp4',
                fps=args.fps,
                dpi=args.dpi
            )
    
    if args.create_gif:
        if args.rotating:
            morph.create_rotating_animation(
                output_dir=args.output,
                output_file='morph_rotating.gif',
                format='gif',
                fps=args.fps,
                dpi=args.dpi
            )
        else:
            morph.create_animation(
                output_dir=args.output,
                output_file='morph_animation.gif',
                format='gif',
                fps=args.fps,
                dpi=args.dpi
            )