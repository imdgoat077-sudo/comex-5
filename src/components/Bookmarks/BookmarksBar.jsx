import React from 'react';
import { 
  Search, BookOpen, Flame, Code, Image, Tv, Globe 
} from 'lucide-react';
import GithubIcon from '../Icons/GithubIcon';
import { DEFAULT_BOOKMARKS } from '../../data/defaultSites';
import './BookmarksBar.css';

const ICON_MAP = {
  Search: Search,
  Github: GithubIcon,
  BookOpen: BookOpen,
  Flame: Flame,
  Code: Code,
  Image: Image,
  Tv: Tv,
  Globe: Globe
};

export default function BookmarksBar({ onNavigate }) {
  return (
    <nav className="bookmarks-bar" aria-label="Bookmarks">
      <div className="bookmarks-scroll">
        {DEFAULT_BOOKMARKS.map((item) => {
          const IconComponent = ICON_MAP[item.icon] || Globe;
          return (
            <button
              key={item.id}
              className="bookmark-chip"
              onClick={() => onNavigate(item.url)}
              title={item.url}
            >
              <IconComponent size={13} />
              <span>{item.title}</span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}
